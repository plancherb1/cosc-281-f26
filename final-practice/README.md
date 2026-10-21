# Final practice

Open [handout.pdf](handout.pdf) for 21 untimed practice questions covering
both halves of the course. This collection is not intended to match the
final exam's length, point distribution, or question order. Part 9(c) and
Question 21 are marked as graduate extensions.

Work through the questions before consulting the separately distributed
solutions. Show calculations and explain your reasoning. There is no
programming submission or autograder for this collection. For additional
RGB/HSV and lighting practice, revisit the PS4 image experiment.

## Writing in LaTeX

The editable [handout.tex](handout.tex) contains the questions and formula
sheet. Put your answers in the `studentanswer` blocks at the
`%%%%% SOLUTION HERE %%%%%` markers. Increase a block's height if you need
more room. The `figures/` folder contains the supplied illustration.

To use Overleaf, upload `handout.tex` and the `figures/` folder together,
preserving the folder name, and select `handout.tex` as the main document.
No other course files or Python environment are needed.

With a local LaTeX installation, run either command from the indicated
directory:

```bash
# From the course repository root:
latexmk -pdf final-practice/handout.tex

# Or from inside final-practice/:
latexmk -pdf handout.tex
```
