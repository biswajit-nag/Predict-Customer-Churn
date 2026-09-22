# Kaggle API setup & leaderboard backfill

The experiment framework can submit a run's bagged test predictions
(`test_proba_mean.npy`) to Kaggle and record the public/private leaderboard
scores in `experiments/runs.csv` (`lb_public` / `lb_private`). This needs the
Kaggle CLI authenticated. The CLI is already installed; you just need to make
your API key available.

---

## 1. Get your API token

1. Go to <https://www.kaggle.com/settings/account> → **API** → **Create New
   Token**. This downloads a `kaggle.json` containing your `username` and `key`.
2. Open it in a text editor — you'll copy the two values in the next step.

---

## 2. Save the key as environment variables (Windows, persistent)

The CLI reads `KAGGLE_USERNAME` and `KAGGLE_KEY` from the environment. Set them
once as **user** environment variables so every future shell (and notebook
kernel) picks them up. In PowerShell:

```powershell
[Environment]::SetEnvironmentVariable('KAGGLE_USERNAME', 'your_kaggle_username', 'User')
[Environment]::SetEnvironmentVariable('KAGGLE_KEY',      'your_kaggle_key',      'User')
```

> `setx KAGGLE_USERNAME "your_kaggle_username"` works too, but truncates values
> over 1024 chars and is more error-prone — prefer the `[Environment]` form above.

**Important:** these take effect in **new** shells only. For your *current*
PowerShell session, also set them live so you don't have to reopen the terminal:

```powershell
$env:KAGGLE_USERNAME = 'your_kaggle_username'
$env:KAGGLE_KEY      = 'your_kaggle_key'
```

For the **Git Bash** shell (the Bash tool), the persistent user vars are
inherited by new sessions automatically; to set them live in an open Bash:

```bash
export KAGGLE_USERNAME='your_kaggle_username'
export KAGGLE_KEY='your_kaggle_key'
```

### Recommended: API token via `kaggle auth login` (no `kaggle.json`)

The modern Kaggle CLI manages an **API token** for you — no manual `kaggle.json`
required. Run once:

```powershell
kaggle auth login
```

This performs a browser auth flow and writes a token to
`C:\Users\<you>\.kaggle\credentials.json`, which the CLI reads automatically on
every call. This is the method this repo is set up to use, and it satisfies
Kaggle's "use API tokens" recommendation without you handling a username+key
file. Re-run `kaggle auth login` if the token expires.

### Legacy alternative: the `kaggle.json` file

Instead of the above you can drop the downloaded `kaggle.json` at
`C:\Users\<you>\.kaggle\kaggle.json`. All three forms work; the framework's
`credentials_available()` recognizes env vars, `credentials.json`, and
`kaggle.json` (honouring `KAGGLE_CONFIG_DIR` if set).

> **Note on environment scope.** A persistent (`User`-scope) env var only reaches
> processes started *after* it was set. If a long-running app (an IDE, a terminal
> session) was open when you set `KAGGLE_USERNAME`/`KAGGLE_KEY`, its child
> shells won't see them until you restart it. The `kaggle auth login` token file
> avoids this entirely — it's read from disk regardless of environment.

---

## 3. Verify

```powershell
kaggle competitions submissions -c playground-series-s6e3
```

If that prints your submission history (or an empty list) without an auth error,
you're set. From Python:

```python
from src.kaggle_io import credentials_available
credentials_available()   # -> True
```

> **Never commit `kaggle.json` or paste your key into code or chat.** It is a
> live credential. `.gitignore` already excludes `.kaggle/`-style secrets; keep
> it that way.

---

## 4. Submit & collect scores

**One run, inline with training** — pass `submit=True` to `save_experiment`:

```python
from src.cv import run_cv_experiment, save_experiment
result = run_cv_experiment(run_config, X_train, y_train, X_test, features)
save_experiment(result, submit=True)   # submits + waits + writes lb_public/lb_private
```

**One already-logged run:**

```python
from src.kaggle_io import submit_run
submit_run("20260610-183508-48286b")   # builds submission.csv, uploads, records score
```

**Backfill every logged run** (resumable, rate-limit aware):

```powershell
python scripts/backfill_lb_scores.py --max 5
```

---

## 5. Daily submission limit — what to expect

Kaggle caps submissions per day (typically **5** for Playground competitions).
There are ~37 full-set runs, so a full backfill spans several days. The backfill
script is built for this:

- it **skips** runs already carrying an `lb_public` score (safe to re-run),
- it submits **best-OOF-AUC first**, so a small daily budget scores the most
  informative runs first,
- it **stops cleanly** when Kaggle reports the daily limit and tells you to
  resume tomorrow.

Just run `python scripts/backfill_lb_scores.py --max 5` once a day until
`experiments/runs.csv` is fully populated.

If the competition has closed, submissions still work as **late submissions**
(they return a public score, and a private score once the private leaderboard is
revealed); if Kaggle rejects them entirely, the CLI error is surfaced verbatim.
