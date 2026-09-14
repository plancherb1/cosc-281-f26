# COSC 81/281: Introduction to Robotics

This repository contains the student materials for the Fall 2026 course. The
course simulator is shared across assignments; each released problem set has
its own top-level folder (`ps0/`, `ps1/`, and so on).

## Repository layout

```text
sim/                 shared course simulator and maps
psN/handout.pdf      problem-set handout
psN/handout.tex      self-contained editable version of the same handout
psN/code/             code to run, complete, and submit
psN/code/tests/       public tests run by code/autograder.py
psN/figures/          figures used in the written submission
```

Open `psN/handout.pdf` first. If you prefer to write in LaTeX, edit the
matching `handout.tex`; it can be uploaded directly to Overleaf. Put answers
inside the provided `studentanswer` environments at the
`%%%%% SOLUTION HERE %%%%%` markers.

## Get the course folder

The easiest graphical option is GitHub Desktop: choose **File > Clone
repository**, select this repository, and choose where to keep the course
folder. When a new problem set is announced, use **Repository > Pull**.

From a terminal, the equivalent commands are:

```bash
git clone git@github.com:plancherb1/cosc-281-f26.git
cd cosc-281-f26
```

You only clone once. Run `git pull` from this folder when a new assignment is
released. Released problem-set folders will normally remain unchanged, so new
assignments can be added without disturbing your earlier work.

Do not make a public fork containing your answers. Students who already use
Git may make local commits or use their own private repository, but neither is
required for the course.

## Install the shared environment once

You need Python 3.12 with Tk GUI support. Tk is part of the operating-system
Python installation, not a package that `pip` can add:

- **Ubuntu/Debian:** run `sudo apt install python3-venv python3-tk`.
- **macOS:** use the Python 3.12 installer from python.org, which includes Tk.
- **Windows:** the python.org installer includes Tk by default; if it is
  missing, modify the Python installation and enable **Tcl/Tk and IDLE**.

Confirm the GUI before continuing. A small demonstration window should open;
close it after the check:

```bash
python3 -m tkinter     # macOS/Linux
python -m tkinter      # Windows
```

Then, from the repository root, create one environment shared by every
assignment.

macOS or Linux:

```bash
python3 -m venv --prompt COSC281 .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows PowerShell:

```powershell
python -m venv --prompt COSC281 .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The `(COSC281)` prefix confirms that the course environment, rather than
another project's environment, is active. Activate it again in each new
terminal. Each handout gives the exact commands and submission files for that
assignment. If setup fails, ask on Slack and include the complete error
message and your operating system.

## Start PS0

Read `ps0/handout.pdf`, then run:

```bash
python ps0/code/autograder.py
```

The environment tests should pass immediately. The implementation tests fail
until you complete the five functions in `ps0/code/warmup.py`.
