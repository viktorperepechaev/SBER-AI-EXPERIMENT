"""Compare a classical Gaussian-copula estimator with published SΩI Fig. 8(d).

Run in this folder:
    python soi_counterexample.py --plot
Requires NumPy and Pillow; --plot additionally requires matplotlib.

SΩI values are digitized from the authors' original embedded figure, NOT a
rerun or raw per-seed prediction. The classical estimator is newly evaluated
on independent draws from the matching distribution and sample budget.
This comparison is selected to demonstrate a counterexample, not to measure
average performance across arbitrary distributions.

Primary sources:
  https://arxiv.org/pdf/2402.05667v3 (Fig. 8(d), p.21; 100k training samples)
  https://github.com/MustaphaBounoua/soi/blob/12952f58136454f813dc2887db220876823e053a/src/benchmark/synthetic_dataset.py
  https://proceedings.mlr.press/v70/singh17a.html
"""

import argparse
import hashlib
import json
from pathlib import Path
from statistics import NormalDist
from time import perf_counter

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent
SEEDS = [11, 22, 33, 55, 77, 0, 1, 2, 3, 4]
RHOS = [.6, .65, .7]
DIM = 20
BLOCKS = [3, 3, 4]
SAMPLES = 100_000
AUTHOR_COMMIT = '12952f58136454f813dc2887db220876823e053a'


def logdet(cov):
    sign, value = np.linalg.slogdet(cov)
    if sign <= 0:
        raise ValueError('Nonpositive covariance determinant')
    return float(value)


def gaussian_omega(cov, dim=DIM):
    precision = np.linalg.inv(cov)
    value = -2 * logdet(cov)
    for start in range(0, len(cov), dim):
        ix = slice(start, start + dim)
        value += logdet(cov[ix, ix]) - logdet(precision[ix, ix])
    return value / 2


def synergy_covariance(rho, dim=DIM):
    """Matches Task_synergy and independent [3,3,4] task combination."""
    base = np.eye(sum(BLOCKS))
    start = 0
    for size in BLOCKS:
        base[start, start+1] = base[start+1, start] = 1 / np.sqrt(size-1)
        for j in range(2, size):
            base[start+1, start+j] = base[start+j, start+1] = rho / np.sqrt(size-1)
        start += size
    return np.kron(base, np.eye(dim))


def copula_estimate(observed, quantiles):
    """Fit full covariance after empirical marginal Gaussianization.

    Does not use the known inverse half-cube transform, true covariance,
    rho, or the independence of the three synthetic subsystems.
    """
    z = np.empty_like(observed)
    for j in range(observed.shape[1]):
        order = np.argsort(observed[:, j])
        z[order, j] = quantiles
    z -= z.mean(axis=0)
    return gaussian_omega(z.T @ z / len(z))


def digitize_panel(panel, truths):
    image = np.asarray(Image.open(panel).convert('RGB')).astype(int)
    assert image.shape[:2] == (907, 1207), 'Use the unmodified original Fig.8(d) image'
    # Original image pixel centers. Labels and positions verified visually.
    x_ticks = np.array([416., 780., 1145.])
    x_values = np.array([-10., -5., 0.])
    y_ticks = np.array([118., 289., 460., 631.])
    y_values = np.array([5., 0., -5., -10.])
    x_to_pixel = np.polyfit(x_values, x_ticks, 1)
    pixel_to_y = np.polyfit(y_ticks, y_values, 1)
    blue = (image[:, :, 2] > image[:, :, 0] + 60) & (image[:, :, 2] > image[:, :, 1] + 25) & (image[:, :, 1] > 60)
    points = []
    for rho, truth in zip(RHOS, truths):
        xp = int(round(np.polyval(x_to_pixel, truth)))
        # Small star-marker neighborhood; pale uncertainty fill is excluded.
        # A small connected line segment remains, so report conservative ±0.15.
        yy, xx = np.where(blue[350:650, xp-7:xp+8])
        if not len(yy):
            raise ValueError('No blue SΩI marker pixels found')
        median_x = float(np.median(xx + xp - 7))
        median_y = float(np.median(yy + 350))
        estimate = float(np.polyval(pixel_to_y, median_y))
        points.append(dict(rho=rho, true_omega_nats=truth, published_soi_approx_nats=estimate,
                           mean_pixel_x=median_x, mean_pixel_y=median_y,
                           mean_abs_error_approx_nats=abs(estimate-truth)))
    return {'method': 'Median saturated-blue pixels near the three leftmost star markers of the unmodified original embedded JPEG; linear calibration from labeled axis ticks.',
            'reading_tolerance_nats': .15,
            'reading_tolerance_note': 'Conservative image-reading tolerance, NOT statistical uncertainty or a confidence interval.',
            'sha256': hashlib.sha256(panel.read_bytes()).hexdigest(),
            'x_tick_pixels': x_ticks.tolist(), 'x_tick_values': x_values.tolist(),
            'y_tick_pixels': y_ticks.tolist(), 'y_tick_values': y_values.tolist(),
            'points': points}


def run():
    quantiles = np.fromiter((NormalDist().inv_cdf((i+.5)/SAMPLES) for i in range(SAMPLES)), float, count=SAMPLES)
    rows, summaries, truths = [], [], []
    for rho in RHOS:
        cov = synergy_covariance(rho)
        truth = gaussian_omega(cov)
        truths.append(truth)
        chol = np.linalg.cholesky(cov)
        estimates, times = [], []
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            latent = rng.standard_normal((SAMPLES, cov.shape[0])) @ chol.T
            observed = latent * np.sqrt(np.abs(latent))
            started = perf_counter()
            estimate = copula_estimate(observed, quantiles)
            elapsed = perf_counter()-started
            estimates.append(estimate)
            times.append(elapsed)
            rows.append(dict(rho=rho, seed=seed, estimate_nats=estimate, error_nats=estimate-truth, fit_seconds=elapsed))
            del latent, observed
        errors = np.array(estimates)-truth
        summaries.append(dict(rho=rho, truth_nats=truth, copula_mean_nats=float(np.mean(estimates)),
                              copula_std_nats=float(np.std(estimates, ddof=1)),
                              copula_rmse_nats=float(np.sqrt(np.mean(errors**2))),
                              copula_max_abs_error_nats=float(np.max(np.abs(errors))),
                              median_fit_seconds=float(np.median(times))))
        print(json.dumps(summaries[-1]), flush=True)
    digitized = digitize_panel(ROOT/'soi_fig8d_original.jpg', truths)
    for result, source in zip(summaries, digitized['points']):
        result['published_soi_approx_nats'] = source['published_soi_approx_nats']
        result['published_soi_mean_abs_error_approx_nats'] = source['mean_abs_error_approx_nats']
        result['image_tolerance_adjusted_soi_mean_error_lower_nats'] = max(0, source['mean_abs_error_approx_nats'] - digitized['reading_tolerance_nats'])
    output = {
        'setup': {'benchmark': 'synergy', 'setting': 0, 'blocks': BLOCKS, 'dimension_per_variable': DIM,
                  'total_coordinates': DIM*sum(BLOCKS), 'samples_for_classical_estimator': SAMPLES,
                  'paper_training_samples': 100_000, 'paper_test_samples': 10_000,
                  'seeds_for_classical_estimator': SEEDS, 'rhos': RHOS,
                  'transformation': 'x * sqrt(abs(x))', 'units': 'nats', 'author_code_commit': AUTHOR_COMMIT,
                  'fit_uses_true_covariance': False, 'fit_uses_inverse_half_cube': False,
                  'network_retrained': False,
                  'sampling': 'Independent normal draws, same covariance law and sample budget; not identical author data or paired runs.',
                  'normalization': 'Positive affine standardization after the half-cube transform leaves the rank estimator exactly unchanged.'},
        'summaries': summaries, 'replicates': rows, 'digitization': digitized,
        'interpretation': 'Compare classical Monte Carlo RMSE with the absolute deviation of the published SΩI mean. These are explicitly different reported statistics; the latter is a lower bound on the empirical RMSE of the author runs represented by that mean. Raw author predictions were unavailable in the repository and no significance test or full neural reproduction is claimed.',
        'scope': 'Selected counterexample within a Gaussian-copula family. Does not show universal inferiority of SΩI or invalidate its theoretical identities.'
    }
    (ROOT/'soi_counterexample_results.json').write_text(json.dumps(output, indent=2)+'\n')
    return output


def plot(result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = result['summaries']
    truth = np.array([r['truth_nats'] for r in rows])
    published = np.array([r['published_soi_approx_nats'] for r in rows])
    classical = np.array([r['copula_mean_nats'] for r in rows])
    std = np.array([r['copula_std_nats'] for r in rows])
    fig, ax = plt.subplots(figsize=(8.6, 6.6), layout='constrained')
    ax.plot([-12.5, -5.5], [-12.5, -5.5], ':', color='.35', label='Точное значение: оценка = истина')
    order = np.argsort(truth)
    ax.plot(truth[order], published[order], '*-', color='#2878a5', markersize=12, label='SΩI: опубликованная средняя, ≈ по рис. 8(d)')
    ax.errorbar(truth[order], classical[order], yerr=std[order], fmt='o-', capsize=3, color='#bb572f', markersize=6, label='Гауссовская копула: наша средняя ± SD, 10 повторов')
    ax.annotate(f'SΩI ≈ {published[-1]:.2f}', xy=(truth[-1], published[-1]), xytext=(-11.3, -8.8), fontsize=11,
                arrowprops={'arrowstyle':'-', 'color':'#2878a5'})
    ax.annotate(f'Копула ≈ {classical[-1]:.3f}\nRMSE = {rows[-1]["copula_rmse_nats"]:.3f}',
                xy=(truth[-1], classical[-1]), xytext=(-10.9, -11.8), fontsize=11,
                arrowprops={'arrowstyle':'-', 'color':'#bb572f'})
    ax.set(xlabel='Истинная O-information, нат', ylabel='Оценённая O-information, нат',
           xlim=(-12.5, -5.5), ylim=(-12.5, -5.0))
    ax.set_title('Контрпример: синергия после нелинейного преобразования\n10 переменных × 20 координат, 100 000 наблюдений', fontsize=13)
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(alpha=.2)
    fig.supxlabel('Источник SΩI: arXiv:2402.05667v3, рис. 8(d), стр. 21.\nКривая оцифрована (допуск ±0,15 нат); сеть заново не обучалась.', fontsize=9)
    fig.savefig(ROOT/'soi_counterexample.png', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--plot-only', action='store_true')
    args = parser.parse_args()
    result = json.loads((ROOT/'soi_counterexample_results.json').read_text()) if args.plot_only else run()
    if args.plot or args.plot_only:
        plot(result)
