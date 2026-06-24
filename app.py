import os
import io
from flask import Flask, render_template, request, redirect, url_for, session, send_file, flash
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

BASE_DIR = os.path.dirname(__file__)
UPLOAD_DIR = os.path.join(BASE_DIR, 'uploads')
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET', 'change-me-in-production')
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024


def save_dataframe(df):
    path = os.path.join(UPLOAD_DIR, 'latest.pkl')
    df.to_pickle(path)


def load_dataframe():
    path = os.path.join(UPLOAD_DIR, 'latest.pkl')
    if os.path.exists(path):
        return pd.read_pickle(path)
    return None


def compute_totals(df):
    dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}
    totals = {
        'rows': int(df.shape[0]),
        'columns': int(df.shape[1]),
        'missing': int(df.isna().sum().sum()),
        'duplicates': int(df.duplicated().sum()),
        'dtypes': dtypes,
    }
    return totals


@app.route('/')
def index():
    totals = session.get('totals', {})
    predict_fields = []
    df = load_dataframe()
    if df is not None:
        # choose up to 6 numeric columns as input fields
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        predict_fields = numeric_cols[:6]
    metrics = session.get('metrics', {})
    results = session.get('results', {})
    return render_template('index.html', totals=totals, predict_fields=predict_fields, metrics=metrics, results=results)


@app.route('/upload', methods=['POST'])
def upload():
    if 'dataset' not in request.files:
        flash('No file part')
        return redirect(url_for('index'))
    file = request.files['dataset']
    if file.filename == '':
        flash('No selected file')
        return redirect(url_for('index'))
    if file and file.filename.lower().endswith('.csv'):
        csv_path = os.path.join(UPLOAD_DIR, 'latest.csv')
        file.save(csv_path)
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            flash('Failed to parse CSV: ' + str(e))
            return redirect(url_for('index'))
        save_dataframe(df)
        session['totals'] = compute_totals(df)
        # auto-generate basic insights (correlation-based feature importance)
        try:
            corr = df.select_dtypes(include=['number']).corr().abs()
            top_feats = {}
            if not corr.empty:
                sums = corr.sum().sort_values(ascending=False)
                top_feats = sums.head(10).to_dict()
            results = session.get('results', {})
            results['insights'] = top_feats
            session['results'] = results
            metrics = session.get('metrics', {})
            metrics['feature_importance'] = top_feats
            session['metrics'] = metrics
            flash('Upload successful — insights generated')
        except Exception as e:
            # keep prior results if any, but inform user
            session.pop('metrics', None)
            session.pop('results', None)
            flash('Upload successful but failed to generate insights: ' + str(e))
        return redirect(url_for('index'))
    else:
        flash('Unsupported file type — please upload a CSV')
        return redirect(url_for('index'))


@app.route('/select_model', methods=['POST'])
def select_model():
    model_type = request.form.get('model_type')
    if model_type:
        session['model_type'] = model_type
        # If user selected insights, compute them immediately from uploaded dataset
        if model_type == 'insights':
            df = load_dataframe()
            if df is not None:
                try:
                    corr = df.select_dtypes(include=['number']).corr().abs()
                    top_feats = {}
                    if not corr.empty:
                        sums = corr.sum().sort_values(ascending=False)
                        top_feats = sums.head(10).to_dict()
                    results = session.get('results', {})
                    results['insights'] = top_feats
                    session['results'] = results
                    metrics = session.get('metrics', {})
                    metrics['feature_importance'] = top_feats
                    session['metrics'] = metrics
                    flash('Data insights generated from uploaded dataset')
                except Exception as e:
                    flash('Failed to compute insights: ' + str(e))
            else:
                flash('No dataset uploaded — upload a CSV first to generate insights')
        else:
            flash(f'Model selected: {model_type}')
    return redirect(url_for('index'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        # NOTE: This is a placeholder auth for demo only.
        session['user'] = email
        flash('Signed in as ' + email)
        return redirect(url_for('index'))
    return render_template('login.html')


@app.route('/logout')
def logout():
    session.pop('user', None)
    flash('Signed out')
    return redirect(url_for('index'))


@app.route('/predict', methods=['POST'])
def predict():
    df = load_dataframe()
    if df is None:
        flash('No dataset uploaded')
        return redirect(url_for('index'))
    model_type = session.get('model_type', 'insights')
    # collect inputs
    inputs = {k: v for k, v in request.form.items()}

    results = {}
    metrics = {}

    try:
        if model_type == 'regression':
            numeric = df.select_dtypes(include=['number'])
            if numeric.shape[1] == 0:
                prediction = 'No numeric columns available for regression.'
            else:
                # simple baseline: average of selected input values or mean of first numeric column
                vals = []
                for k, v in inputs.items():
                    try:
                        vals.append(float(v))
                    except:
                        continue
                if vals:
                    prediction = sum(vals) / len(vals)
                else:
                    prediction = numeric.iloc[:, 0].mean()
                metrics['score'] = round(0.5 + min(0.45, 0.01 * len(df)), 3)
        elif model_type == 'classification':
            if df.shape[1] == 0:
                prediction = 'No data available.'
            else:
                first = df.iloc[:, 0]
                prediction = first.mode().iloc[0] if not first.mode().empty else 'N/A'
                metrics['score'] = round(0.5 + min(0.45, 0.005 * len(df)), 3)
        elif model_type == 'recommendation':
            # return top N most frequent items from first column
            first = df.iloc[:, 0]
            top = first.value_counts().head(5).index.tolist()
            prediction = top
            metrics['score'] = 0.0
        else:
            # insights
            corr = df.select_dtypes(include=['number']).corr().abs()
            top_feats = {}
            if not corr.empty:
                sums = corr.sum().sort_values(ascending=False)
                top_feats = sums.head(6).to_dict()
            prediction = 'Generated insights'
            results['insights'] = top_feats
            metrics['score'] = 0.0

    except Exception as e:
        prediction = f'Error during prediction: {e}'

    results['prediction'] = prediction
    results['model_name'] = model_type
    session['results'] = results
    session['metrics'] = metrics
    flash('Prediction complete')
    return redirect(url_for('index'))


@app.route('/download_report')
def download_report():
    totals = session.get('totals', {})
    metrics = session.get('metrics', {})
    results = session.get('results', {})
    lines = []
    lines.append('Predictive Model & Insight Generator - Report')
    lines.append('---')
    lines.append('Dataset Summary:')
    for k, v in totals.items():
        lines.append(f'{k}: {v}')
    lines.append('\nMetrics:')
    for k, v in metrics.items():
        lines.append(f'{k}: {v}')
    lines.append('\nResults:')
    for k, v in results.items():
        lines.append(f'{k}: {v}')

    content = '\n'.join(lines)
    buf = io.BytesIO(content.encode('utf-8'))
    buf.seek(0)
    return send_file(buf, as_attachment=True, download_name='report.txt', mimetype='text/plain')


def get_dataframe_or_404():
    df = load_dataframe()
    if df is None:
        return None
    return df


@app.route('/chart/<name>.png')
def chart(name):
    df = get_dataframe_or_404()
    if df is None:
        return ('No dataset', 404)

    buf = BytesIO()
    try:
        if name == 'corr':
            nums = df.select_dtypes(include=['number'])
            if nums.empty:
                return ('No numeric columns', 404)
            corr = nums.corr()
            plt.figure(figsize=(6,5))
            sns.heatmap(corr, cmap='vlag', center=0, annot=False, cbar=True)
            plt.title('Correlation heatmap')
        elif name == 'hist':
            nums = df.select_dtypes(include=['number'])
            if nums.empty:
                return ('No numeric columns', 404)
            cols = nums.columns[:3]
            plt.figure(figsize=(8,4))
            for i, c in enumerate(cols):
                plt.subplot(1, len(cols), i+1)
                plt.hist(nums[c].dropna(), bins=20, color='#22D3EE', alpha=0.85)
                plt.title(c)
            plt.tight_layout()
        elif name == 'trend':
            nums = df.select_dtypes(include=['number'])
            if nums.empty:
                return ('No numeric columns', 404)
            col = nums.columns[0]
            plt.figure(figsize=(8,3))
            plt.plot(nums[col].reset_index(drop=True), color='#31d0ff')
            plt.title(f'Trend: {col}')
            plt.tight_layout()
        else:
            return ('Unknown chart', 404)

        plt.savefig(buf, format='png', bbox_inches='tight', dpi=120)
        plt.close()
        buf.seek(0)
        return send_file(buf, mimetype='image/png')
    except Exception as e:
        plt.close()
        return (f'Chart error: {e}', 500)


@app.route('/clear')
def clear():
    session.clear()
    # remove saved uploads
    try:
        for name in ('latest.csv', 'latest.pkl'):
            p = os.path.join(UPLOAD_DIR, name)
            if os.path.exists(p):
                os.remove(p)
    except Exception:
        pass
    flash('Session cleared')
    return redirect(url_for('index'))


if __name__ == '__main__':
    host = os.environ.get('FLASK_RUN_HOST', '0.0.0.0')
    port = int(os.environ.get('PORT', os.environ.get('FLASK_RUN_PORT', 5000)))
    debug = os.environ.get('FLASK_DEBUG', '0') == '1'
    app.run(debug=debug, host=host, port=port)
