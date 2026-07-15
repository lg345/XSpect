"""
Modify mfx100895324_static_xes_visualization.ipynb so the Fe(III) reference
spectrum is a COMBINATION of the ferricyanide runs (36, 37, 38) instead of a
single run (36).

Combination method: sum raw counts across the reference runs, then
area-normalize -- identical to how the reduced Fe(II) end-member (runs 39-42)
is already built in the LCF, so both end-members are photon-weighted the same
way.

Touches the three cells that consume the reference:
  cell 3  -- define REFERENCE_RUNS + keep REF_RUN for labels
  cell 7  -- add combined_ref() helper next to area_norm()
  cell 15 -- IAD uses combined reference
  cell 17 -- LCF oxidized end-member uses combined reference
  cell 19 -- pointwise % deviation uses combined reference
"""

import nbformat

PATH = "mfx100895324_static_xes_visualization.ipynb"
nb = nbformat.read(PATH, as_version=4)


def set_cell(idx, text):
    nb.cells[idx]["source"] = text
    # clear stale outputs/exec counts; the notebook will be re-executed
    if nb.cells[idx].get("cell_type") == "code":
        nb.cells[idx]["outputs"] = []
        nb.cells[idx]["execution_count"] = None


# --------------------------------------------------------------------------- #
# Cell 3 -- reference / end-member run definitions
# --------------------------------------------------------------------------- #
set_cell(
    3,
    r"""# Sample composition loaded from the run manifest (runs.csv) -- the single
# source of truth documented in README.md. reduced_fraction = nominal Fe(II)
# ferrocyanide fraction (0 = pure ferricyanide reference, 1 = pure ferrocyanide).
# ocv_mV = open-circuit voltage recorded for the sample well (where noted).
# Run 47 is metallic Fe foil (calibration standard), NOT a cyanide sample, so
# it is excluded from all titration/deviation analysis and flagged separately.
# (Full beamtime record incl. Shift-2 in-situ echem is in BEAMTIME_MANIFEST.md.)
import csv

sample_label = {}
reduced_fraction = {}
ocv_mV = {}
role = {}
with open(os.path.join(HERE, 'runs.csv')) as fh:
    for row in csv.DictReader(fh):
        rn = int(row['run'])
        sample_label[rn] = row['sample_label']
        role[rn] = row['role']
        if row['reduced_fraction'] != '':
            reduced_fraction[rn] = float(row['reduced_fraction'])
        if row.get('ocv_mV', '') != '':
            ocv_mV[rn] = float(row['ocv_mV'])

FOIL_RUNS = [rn for rn, r in role.items() if r == 'excluded']   # Fe foil, excluded
chem_runs = [rn for rn in run_nums if rn not in FOIL_RUNS]      # cyanide samples only

# Oxidized reference = ALL ferricyanide runs (nominal Fe(II) fraction = 0),
# combined into one low-noise reference spectrum. This mirrors the reduced
# Fe(II) end-member (runs 39-42) used by the LCF, so both end-members are
# built the same way (sum raw counts -> area-normalize; see combined_ref()).
REFERENCE_RUNS = [rn for rn in run_nums
                  if reduced_fraction.get(rn, None) == 0.0 and rn not in FOIL_RUNS]
REF_RUN = REFERENCE_RUNS[0]   # representative single run, kept for plot labels

print(f'Reference (0% reduced): runs {REFERENCE_RUNS}  (combined)')
print(f'Excluded (Fe foil):     runs {FOIL_RUNS}')
print(f'Cyanide runs analyzed:  {chem_runs}')""",
)


# --------------------------------------------------------------------------- #
# Cell 7 -- add combined_ref() helper alongside area_norm()
# --------------------------------------------------------------------------- #
set_cell(
    7,
    r"""def area_norm(y):
    y = np.asarray(y, dtype=float)
    tot = np.nansum(y)
    return y / tot if tot > 0 else y

def combined_ref(spectra, ref_runs=None):
    """
    + '"""'
    + r"""Area-normalized reference from a COMBINATION of runs.

    Sums the raw per-run spectra (photon-weighted, so high-statistics runs
    contribute more) and then area-normalizes -- identical to how the reduced
    Fe(II) end-member is built in the LCF. Defaults to REFERENCE_RUNS
    (the ferricyanide / 0% reduced runs).
    """
    + '"""'
    + r"""
    if ref_runs is None:
        ref_runs = REFERENCE_RUNS
    return area_norm(np.nansum([spectra[rn] for rn in ref_runs], axis=0))

def per_run_grid(spectra, energy, ref_lines, title, color):
    n = len(spectra)
    ncol = 4
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 2.6 * nrow),
                             sharex=True, squeeze=False)
    for ax in axes.flat:
        ax.set_visible(False)
    for i, rn in enumerate(sorted(spectra)):
        ax = axes.flat[i]
        ax.set_visible(True)
        ax.plot(energy, area_norm(spectra[rn]), color=color, lw=1)
        for e in ref_lines:
            ax.axvline(e, color='gray', ls=':', lw=0.8)
        ax.set_title(f'run {rn}  (n={shots[rn]})', fontsize=9)
        ax.tick_params(labelsize=7)
    fig.suptitle(title, y=1.005)
    fig.supxlabel('Emission energy (eV)')
    fig.supylabel('Norm. intensity')
    plt.tight_layout()
    plt.show()

per_run_grid(ka, ka_energy, [6404, 6391],
             'Fe K\u03b1 XES per run (area-normalized) \u2014 mfx100895324', 'C3')""",
)


# --------------------------------------------------------------------------- #
# Cell 15 -- IAD from combined reference
# --------------------------------------------------------------------------- #
set_cell(
    15,
    r"""def iad(spectra, runs_list, ref=None):
    if ref is None:
        ref = combined_ref(spectra)          # combined ferricyanide reference
    return {rn: float(np.nansum(np.abs(area_norm(spectra[rn]) - ref))) for rn in runs_list}

# Combined reference (runs 36-38) computed once per line, reused for every run.
ref_ka = combined_ref(ka)
ref_kb = combined_ref(kb)

# IAD for all runs (incl. foil) so we can show the foil as an outlier
iad_ka = iad(ka, run_nums, ref=ref_ka)
iad_kb = iad(kb, run_nums, ref=ref_kb)

fig, axes = plt.subplots(1, 2, figsize=(15, 5))
for ax, iadv, lbl, col in [(axes[0], iad_ka, 'K\u03b1', 'C3'),
                           (axes[1], iad_kb, 'K\u03b2', 'C0')]:
    # nominal reduced fraction on x for cyanide runs; foil plotted at x=-0.1
    xs = [reduced_fraction[rn] for rn in chem_runs]
    ys = [100 * iadv[rn] for rn in chem_runs]
    ax.scatter(xs, ys, c=col, s=45, zorder=3)
    for rn in chem_runs:
        ax.annotate(str(rn), (reduced_fraction[rn], 100 * iadv[rn]),
                    fontsize=7, xytext=(3, 3), textcoords='offset points')
    for rn in FOIL_RUNS:
        ax.scatter([-0.12], [100 * iadv[rn]], marker='x', c='k', s=60, zorder=3)
        ax.annotate(f'foil {rn}', (-0.12, 100 * iadv[rn]), fontsize=7,
                    xytext=(3, 3), textcoords='offset points')
    ax.set_xlabel('Nominal Fe(II) fraction'); ax.set_ylabel('IAD \u00d7 100 (% spectral change)')
    ax.set_title(f'{lbl}: IAD vs nominal reduction'); ax.grid(alpha=0.3)
plt.tight_layout(); plt.show()

print(f'Reference = combined ferricyanide runs {REFERENCE_RUNS}')
print('run  label       nominal   IAD_K\u03b1%   IAD_K\u03b2%')
for rn in run_nums:
    tag = '(foil)' if rn in FOIL_RUNS else f'{reduced_fraction.get(rn, np.nan):.2f}'
    print(f'{rn:3d}  {sample_label[rn]:10s}  {tag:>7s}   {100*iad_ka[rn]:6.2f}   {100*iad_kb[rn]:6.2f}')""",
)


# --------------------------------------------------------------------------- #
# Cell 17 -- LCF oxidized end-member from combined reference
# --------------------------------------------------------------------------- #
set_cell(
    17,
    r"""from numpy.linalg import lstsq

REDUCED_RUNS = [39, 40, 41, 42]   # pure Fe(II) ferrocyanide end-member

def lcf_fraction(spectra, energy):
    # Both end-members built the same way: sum raw counts -> area-normalize.
    ref = combined_ref(spectra, REFERENCE_RUNS)   # oxidized Fe(III), runs 36-38
    red = combined_ref(spectra, REDUCED_RUNS)     # reduced  Fe(II),  runs 39-42
    # Model matrix columns: [reduced, oxidized]; fit S = f*red + (1-f)*ref
    # => S - ref = f*(red - ref)  -> single-parameter least squares
    basis = (red - ref)
    fracs = {}
    for rn in chem_runs:
        y = area_norm(spectra[rn]) - ref
        f = float(np.dot(basis, y) / np.dot(basis, basis))
        fracs[rn] = f
    return fracs

f_ka = lcf_fraction(ka, ka_energy)
f_kb = lcf_fraction(kb, kb_energy)

fig, ax = plt.subplots(figsize=(7, 7))
for fv, lbl, col in [(f_ka, 'K\u03b1', 'C3'), (f_kb, 'K\u03b2', 'C0')]:
    xs = [reduced_fraction[rn] for rn in chem_runs]
    ys = [fv[rn] for rn in chem_runs]
    ax.scatter(xs, ys, c=col, s=45, label=lbl, zorder=3)
ax.plot([0, 1], [0, 1], 'k--', lw=1, label='ideal (fitted = nominal)')
ax.set_xlabel('Nominal Fe(II) fraction'); ax.set_ylabel('LCF fitted fraction reduced')
ax.set_title('Fraction reduced: LCF vs nominal'); ax.legend(); ax.grid(alpha=0.3)
ax.set_aspect('equal'); plt.tight_layout(); plt.show()

print(f'Oxidized end-member = combined runs {REFERENCE_RUNS}; '
      f'reduced end-member = combined runs {REDUCED_RUNS}')
print('run  label       nominal   f_K\u03b1     f_K\u03b2')
for rn in chem_runs:
    print(f'{rn:3d}  {sample_label[rn]:10s}  {reduced_fraction[rn]:6.2f}   '
          f'{100*f_ka[rn]:5.1f}%   {100*f_kb[rn]:5.1f}%')""",
)


# --------------------------------------------------------------------------- #
# Cell 19 -- pointwise % deviation from combined reference
# --------------------------------------------------------------------------- #
set_cell(
    19,
    r"""def pct_dev(spectra, energy, eps_frac=1e-3):
    ref = combined_ref(spectra)                        # combined ferricyanide reference
    floor = ref.max() * eps_frac                       # avoid blow-up in the tails
    safe = np.where(ref > floor, ref, np.nan)
    out = {}
    for rn in chem_runs:
        out[rn] = 100 * (area_norm(spectra[rn]) - ref) / safe
    return out

# one representative run per nominal composition (first of each group)
rep = {}
for rn in chem_runs:
    rep.setdefault(reduced_fraction[rn], rn)
rep_runs = [rep[k] for k in sorted(rep)]

pd_ka = pct_dev(ka, ka_energy)
pd_kb = pct_dev(kb, kb_energy)

fig, axes = plt.subplots(1, 2, figsize=(15, 5))
for ax, pdv, energy, lbl, refl in [
        (axes[0], pd_ka, ka_energy, 'K\u03b1', (6404, 6391)),
        (axes[1], pd_kb, kb_energy, 'K\u03b2', (7058,))]:
    for rn in rep_runs:
        ax.plot(energy, pdv[rn], lw=1.2,
                label=f'run {rn} ({sample_label[rn]}, {reduced_fraction[rn]:.0%})')
    ax.axhline(0, color='gray', lw=0.8)
    for e in refl:
        ax.axvline(e, color='gray', ls=':', lw=0.8)
    ax.set_xlabel('Emission energy (eV)'); ax.set_ylabel('% deviation from ref')
    ax.set_title(f'{lbl}: pointwise % deviation'); ax.legend(fontsize=8)
plt.tight_layout(); plt.show()""",
)


# --------------------------------------------------------------------------- #
# Cell 16 markdown -- update the LCF description to say "combined"
# --------------------------------------------------------------------------- #
nb.cells[16]["source"] = (
    "## 7. Fraction reduced \u2014 two-component linear-combination fit (LCF)\n"
    "\n"
    "Fit each cyanide spectrum as a mixture of the oxidized reference and the\n"
    "fully-reduced Fe(II) end-member:\n"
    "\n"
    "$$S_i(E) = f_i\\, S_{\\mathrm{reduced}}(E) + (1-f_i)\\, S_{\\mathrm{ref}}(E)$$\n"
    "\n"
    "The fitted $f_i$ **is** the fraction reduced. Both end-members are built the\n"
    "same way \u2014 raw counts summed across runs, then area-normalized: the\n"
    "**oxidized reference = ferricyanide runs 36\u201338** and the **reduced\n"
    "end-member = ferrocyanide runs 39\u201342**. Combining the ferricyanide runs\n"
    "gives a lower-noise reference and makes the two end-members symmetric.\n"
    "Because the mixtures have known nominal compositions, plotting fitted vs\n"
    "nominal fraction validates the method (ideal = the diagonal). Run 47 (foil)\n"
    "is excluded."
)

# Cell 14 markdown -- IAD: mention combined reference
nb.cells[14]["source"] = (
    "## 6. Percent deviation from reference \u2014 IAD\n"
    "\n"
    "**Integrated Absolute Difference** (Vank\u00f3 et al.), the standard scalar metric\n"
    "for K\u03b2 XES redox series. Each spectrum is area-normalized, then\n"
    "\n"
    "$$\\mathrm{IAD}_i = \\sum_E \\lvert S_i(E) - S_{\\mathrm{ref}}(E)\\rvert$$\n"
    "\n"
    "Because area-normalized spectra integrate to 1, IAD is a fractional change;\n"
    "\u00d7100 gives **percent spectral change** vs the ferricyanide reference. The\n"
    "reference is the **combined ferricyanide runs (36\u201338)** \u2014 raw counts summed,\n"
    "then area-normalized \u2014 for lower noise. IAD is monotonic with degree of\n"
    "reduction, so it should track the nominal Fe(II) fraction. Computed for both\n"
    "K\u03b1 and K\u03b2. Run 47 (Fe foil) is shown separately as a sanity check \u2014 it\n"
    "should be a large outlier."
)

nbformat.write(nb, PATH)
print("Updated", PATH)
