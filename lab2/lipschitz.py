import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import time

def make_function_from_string(func_str):
    s = func_str.strip()
    if '=' in s:
        s = s.split('=', 1)[1].strip()
    expr = s
    import numpy as _np
    safe = {
        'np': _np, 'sin': _np.sin, 'cos': _np.cos, 'tan': _np.tan,
        'exp': _np.exp, 'sqrt': _np.sqrt, 'log': _np.log, 'log10': _np.log10,
        'abs': _np.abs, 'arctan': _np.arctan, 'arcsin': _np.arcsin, 'arccos': _np.arccos,
        'pi': _np.pi, 'e': _np.e, 'pow': _np.power, 'sign': _np.sign,
        'floor': _np.floor, 'ceil': _np.ceil,
    }
    def f(x):
        local = {'x': x}
        return eval(expr, {"__builtins__": None}, {**safe, **local})
    return f, expr

def lipschitz_global_minimize(f, a, b, eps=1e-2, L=None, max_iter=10000):
    t0 = time.time()
    mid = 0.5*(a+b)
    xs = [a, mid, b]
    fs = [float(f(a)), float(f(mid)), float(f(b))]
    best_idx = int(np.argmin(fs))
    x_best = xs[best_idx]
    f_best = fs[best_idx]
    if L is None:
        max_slope = 0.0
        for i in range(len(xs)):
            for j in range(i+1, len(xs)):
                dx = abs(xs[i] - xs[j])
                if dx > 1e-15:
                    slope = abs(fs[i] - fs[j]) / dx
                    if slope > max_slope:
                        max_slope = slope
        L = max(1e-6, max_slope * 1.2)
    safety = 1.5
    history = list(zip(xs, fs))
    it = 0
    while it < max_iter:
        it += 1
        order = np.argsort(xs)
        xs_ord = [xs[i] for i in order]
        fs_ord = [fs[i] for i in order]
        candidates = []
        phis = []
        for i in range(len(xs_ord) - 1):
            x_i, x_j = xs_ord[i], xs_ord[i+1]
            f_i, f_j = fs_ord[i], fs_ord[i+1]
            if L <= 0:
                x_star = 0.5*(x_i + x_j)
            else:
                x_star = ((f_i - f_j) + L*(x_i + x_j)) / (2.0 * L)
            x_star = max(min(x_star, x_j), x_i)
            phi_val = -1e300
            for k in range(len(xs)):
                val = fs[k] - L * abs(x_star - xs[k])
                if val > phi_val:
                    phi_val = val
            candidates.append((x_star, phi_val, x_i, x_j))
            phis.append(phi_val)
        min_phi = min(phis)
        idx_min = int(np.argmin(phis))
        x_next = candidates[idx_min][0]
        if x_next in xs:
            found_new = False
            for i in range(len(xs_ord) - 1):
                alt = 0.5*(xs_ord[i] + xs_ord[i+1])
                if alt not in xs:
                    x_next = alt
                    found_new = True
                    break
            if not found_new:
                time_elapsed = time.time() - t0
                history = list(zip(xs, fs))
                return x_best, f_best, it-1, history, min_phi, time_elapsed
        if (f_best - min_phi) <= eps:
            time_elapsed = time.time() - t0
            history = list(zip(xs, fs))
            return x_best, f_best, it-1, history, min_phi, time_elapsed
        f_next = float(f(x_next))
        xs.append(x_next)
        fs.append(f_next)
        history.append((x_next, f_next))
        if f_next < f_best:
            f_best = f_next
            x_best = x_next
        max_slope = 0.0
        for i in range(len(xs)):
            for j in range(i+1, len(xs)):
                dx = abs(xs[i] - xs[j])
                if dx > 1e-15:
                    slope = abs(fs[i] - fs[j]) / dx
                    if slope > max_slope:
                        max_slope = slope
        if max_slope * 1.0001 > L:
            L = max(L, max_slope * safety)
    time_elapsed = time.time() - t0
    history = list(zip(xs, fs))
    return x_best, f_best, it, history, None, time_elapsed

def plot_function_and_envelope(f, expr, a, b, history, L, x_best, f_best, save_prefix='results'):
    xs_grid = np.linspace(a, b, 2000)
    ys = np.array([float(f(x)) for x in xs_grid])
    xs_sample = [p[0] for p in history]
    fs_sample = [p[1] for p in history]
    phi_vals = np.full_like(xs_grid, -np.inf, dtype=float)
    for xi, fi in zip(xs_sample, fs_sample):
        phi_vals = np.maximum(phi_vals, fi - L * np.abs(xs_grid - xi))
    plt.figure(figsize=(10,6))
    plt.plot(xs_grid, ys, label='f(x) = ' + expr, linewidth=1.2)
    plt.scatter(xs_sample, fs_sample, c='red', s=25, label='evaluated points')
    plt.plot(xs_grid, phi_vals, linestyle='--', linewidth=1.2, label='lower envelope phi(x)')
    plt.scatter([x_best], [f_best], c='green', s=60, label=f'Found min ≈ ({x_best:.6g}, {f_best:.6g})')
    plt.axvline(a, color='gray', linestyle=':', linewidth=0.8)
    plt.axvline(b, color='gray', linestyle=':', linewidth=0.8)
    plt.legend()
    plt.title('Function and Lipschitz lower envelope')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.grid(True, linestyle=':')
    fname = save_prefix + '_function.png'
    plt.savefig(fname, dpi=200)
    plt.close()
    plt.figure(figsize=(10,6))
    for xi, fi in zip(xs_sample, fs_sample):
        v = fi - L * np.abs(xs_grid - xi)
        plt.plot(xs_grid, v, alpha=0.4)
    plt.plot(xs_grid, phi_vals, linestyle='--', linewidth=1.5, label='phi(x)')
    plt.scatter(xs_sample, fs_sample, c='red', s=25)
    plt.scatter([x_best], [f_best], c='green', s=60)
    plt.title('V-functions and envelope')
    plt.xlabel('x')
    plt.ylabel('value')
    plt.grid(True, linestyle=':')
    plt.legend()
    fname2 = save_prefix + '_envelope.png'
    plt.savefig(fname2, dpi=200)
    plt.close()
    return fname, fname2

def save_report_pdf(save_pdf_path, func_str, a, b, eps, L_used, x_best, f_best, iterations, time_elapsed, image_files):
    with PdfPages(save_pdf_path) as pdf:
        fig_text = plt.figure(figsize=(8.27, 11.69))
        fig_text.clf()
        txt = fig_text.text(0.02, 0.98, f'Global minimization report\n', fontsize=14, weight='bold')
        text_body = f'''
Function: f(x) = {func_str}
Interval: [{a}, {b}]
Requested accuracy eps: {eps}
Used Lipschitz constant L (estimate): {L_used}

Found approximate minimizer:
  x_min ≈ {x_best}
  f(x_min) ≈ {f_best}

Iterations: {iterations}
Time elapsed: {time_elapsed:.4f} s
'''
        fig_text.text(0.02, 0.88, text_body, fontsize=10, va='top')
        pdf.savefig(fig_text)
        plt.close(fig_text)
        for img in image_files:
            fig = plt.figure(figsize=(8.27, 11.69))
            plt.axis('off')
            im = plt.imread(img)
            plt.imshow(im)
            pdf.savefig(fig)
            plt.close(fig)
    return save_pdf_path

if __name__ == '__main__':
    func_input = "10 + x**2 - 10*np.cos(2*np.pi*x)"
    a = -5.12
    b = 5.12
    eps = 0.01

    func_input = "np.sin(3*np.pi*x)**2 + (x-1)**2*(1 + np.sin(2*np.pi*x)**2)"
    a = -10
    b = 10
    eps = 0.01

    my_L = None
    f, expr = make_function_from_string(func_input)
    x_min, f_min, iters, history, lower_bound, time_elapsed = lipschitz_global_minimize(
        f, a, b, eps=eps, L=my_L, max_iter=5000
    )
    xs_sample = [p[0] for p in history]
    fs_sample = [p[1] for p in history]
    max_slope = 0.0
    for i in range(len(xs_sample)):
        for j in range(i+1, len(xs_sample)):
            dx = abs(xs_sample[i] - xs_sample[j])
            if dx > 1e-15:
                slope = abs(fs_sample[i] - fs_sample[j]) / dx
                if slope > max_slope:
                    max_slope = slope
    L_used = max_slope * 1.5 if max_slope > 0 else 1.0
    img1, img2 = plot_function_and_envelope(f, expr, a, b, history, L_used, x_min, f_min, save_prefix='results')
    pdf_path = save_report_pdf('results_report.pdf', expr, a, b, eps, L_used, x_min, f_min, iters, time_elapsed, [img1, img2])
    print("Function:", expr)
    print(f"Interval: [{a}, {b}]")
    print("eps:", eps)
    print(f"x_min ≈ {x_min:.8g}")
    print(f"f(x_min) ≈ {f_min:.8g}")
    print("Iterations:", iters)
    print("Lower bound:", lower_bound)
    print(f"Time: {time_elapsed:.4f} s")
    print("Saved PDF:", pdf_path)
    print(L_used)
