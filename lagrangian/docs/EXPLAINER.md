# How This Program Thinks About Bending Metal

*An explanation of plasticity and the computational method behind `PlasticityFEM`,
for a curious, smart reader who is not an engineer.*

---

## 1. The everyday phenomenon

Take a metal paperclip. Bend it a little and let go — it springs back. Bend it
*a lot* and let go — it stays bent. Somewhere between "a little" and "a lot" the
metal crossed a line: below the line it behaves like a spring (deform, release,
return); above the line it takes a **permanent set**.

That permanent set is **plasticity**. Almost everything made of metal — a car
fender, a bridge, a turbine blade, a soda can — lives near or across that line at
some point, and engineers desperately want to predict *what happens when it does*:
how much load before it permanently deforms, where it deforms, whether it tears.

This program is a tool for answering exactly that, for an arbitrary 3D shape, on a
computer, before anything is built. Below I'll explain (a) the physics it
encodes, and (b) the surprisingly deep computational machinery needed to solve it.

---

## 2. The physics, built up in five ideas

### Idea 1 — Stress and strain

Two quantities describe a deforming solid at every point:

- **Strain** is *how much the material has stretched*, locally — a fractional
  change in length. Stretch a 1-meter bar to 1.001 meters and its strain is 0.001
  (0.1%). Strain is geometry; it's dimensionless.
- **Stress** is *how hard the material is being pulled*, locally — force per unit
  area, like pressure. It's what the internal "fibers" of the material feel.

In 3D these aren't single numbers but little tables (tensors) capturing
stretching and shearing in every direction, but the intuition — *strain = how much
it moved, stress = how hard it's pulled* — carries you a long way.

### Idea 2 — The elastic spring (small loads)

For small loads, stress and strain are simply **proportional**: push twice as
hard, it stretches twice as far, and it returns to its original shape when you let
go. This is **elasticity** (Hooke's law — the physics of a spring). The constant
of proportionality is the material's *stiffness*. This regime is reversible: no
memory, no permanent change.

### Idea 3 — The yield surface (the line you can't un-cross)

Every material has a **yield stress**: a threshold of internal "pull" beyond which
it stops behaving like a spring and starts to *flow* permanently. Below yield,
elastic; above yield, plastic.

In 3D the threshold isn't a single number but a **surface** in the space of
possible stress states — picture a balloon. As long as the stress state stays
*inside* the balloon, the material is elastic. The stress state is **not allowed
to go outside** the balloon. If a load tries to push it out, the material yields:
it deforms permanently just enough to keep the stress state *on the surface* of
the balloon. The specific balloon this program uses is the **von Mises** surface
(a cylinder, in the right coordinates), which captures the experimental fact that
*squeezing a metal uniformly from all sides doesn't make it yield* — only
*distortion* (shape change) does. Hydrostatic pressure alone won't permanently
deform steel; twisting and shearing it will.

### Idea 4 — Hardening (the balloon can grow and drift)

If metals simply yielded at a fixed threshold forever, a bent paperclip would be
as easy to keep bending as to start. But real metals get **stronger as you work
them** (try un-bending and re-bending that paperclip in the same spot — it
resists, then snaps). Two flavors, both included here:

- **Isotropic hardening** — the balloon *inflates*: the yield threshold rises
  uniformly as the material accumulates plastic deformation. It gets harder to
  yield in *any* direction.
- **Kinematic hardening** — the balloon *drifts* in the direction you're pushing.
  This produces the **Bauschinger effect**: after you've yielded a metal in
  tension, it yields *more easily* in compression. (Bend the paperclip one way and
  the reverse bend comes suspiciously easy.) This program reproduces that exactly.

### Idea 5 — Path dependence (the crucial complication)

Here is the property that makes plasticity genuinely hard, mathematically and
computationally. An elastic spring only cares about *where it is now* — stretch
it to length L and the stress is the same no matter how you got there. A plastic
material **remembers its history**. The permanent deformation it carries depends
on the *entire path* of loading it experienced, not just the final load.

This is why the program can't just "solve for the answer." It has to **replay the
loading as a sequence of small steps**, carrying the material's memory (its
accumulated permanent strain) forward from each step to the next. Apply the load
in one big jump and you'd get the wrong answer; you must walk up to it.

---

## 3. From a continuous solid to a finite computation

The laws above hold at *every point* of a continuous object — infinitely many
points. A computer can't track infinitely many. The **Finite Element Method
(FEM)** is the bridge.

**The core trick:** chop the object into a mesh of small, simple blocks
("elements") — here, little **bricks** (8-cornered hexahedra). Inside each brick,
pretend the deformation varies in a simple, smooth way pinned to the positions of
its 8 corners. Now the unknowns aren't "the deformation everywhere" (infinite) but
just "how far did each corner move" (finite). A modest model has thousands of
corners; the large ones we'll discuss have **millions**.

Think of it like approximating a smooth curved surface with a faceted 3D model in
a video game: enough small flat facets and it looks smooth. Enough small elements
and the blocky approximation captures the real physics to whatever accuracy you
want.

Each element contributes a little relationship between the forces at its corners
and the movements of its corners. Stitching all elements together (corners shared
between neighboring bricks must agree) produces one enormous system of equations:

> *Given the loads and the parts that are held fixed, find the position of every
> corner such that every internal force balances.*

For an elastic spring-like material this is one big linear system — solvable in
one shot. For plasticity it is **nonlinear** (the stiffness itself changes as the
material yields) **and path-dependent** (Idea 5), so we need the next two pieces.

---

## 4. Two nested loops: walking the load up, and finding balance at each step

### The outer loop — load stepping

Because the material remembers its path, the program applies the total load in a
series of increments (say, 1%, 2%, … 100%). At each increment it finds the new
balanced shape, then **commits** the material's updated memory before moving on.
This is the computational echo of "you must walk up to the load, not jump."

### The inner loop — Newton's method

At each load step, "find the balanced shape" means solving a nonlinear equation.
The program uses **Newton's method**, the same idea you may know for finding where
a curve crosses zero: guess, measure how far off you are (the *residual* — the
amount by which forces don't yet balance), use the local slope to compute a
correction, repeat. Each iteration roughly *squares* the error — wrong by 0.1,
then 0.01, then 0.0001 — so it converges in a handful of steps. (This rapid,
"quadratic" convergence depends on computing the slope *exactly*; getting that
slope right for plasticity is a subtle piece of the math, called the *consistent
tangent*, and the test suite specifically checks that the convergence really is
quadratic.)

### The heart: "return mapping"

The single most important kernel runs at every integration point inside every
element, at every iteration. It answers one question:

> *Given the total strain right now and the material's remembered state, what is
> the stress, and how much new permanent deformation just occurred?*

The algorithm — **radial return mapping** — is beautifully geometric:

1. **Guess elastic.** Assume the step was purely springy and compute a trial
   stress. (Cheap, and usually right for most of the object.)
2. **Check the balloon.** Is the trial stress inside the yield surface? If yes —
   done, the material was elastic here.
3. **If it pokes outside — pull it back.** Project the stress radially back onto
   the (possibly inflated, possibly drifted) yield surface. The amount you had to
   pull it back *is* the new permanent deformation, and it updates the material's
   memory.

That "pull it back onto the surface" is the *return* in return mapping. It is the
numerical embodiment of "the stress state is not allowed to leave the balloon; the
material flows just enough to stay on it." This program does this in a closed form
(an exact formula) because its hardening laws are linear — fast and exact.

---

## 5. The real challenge: doing this 10 million times at once

A small model (a few thousand corners) is easy. The hard, interesting engineering
is at **scale**: a finely-detailed part can need **~10 million unknowns**. The
system of equations is then a 10-million-by-10-million matrix. Most of this
project's recent work was making that tractable on a single ordinary (64 GB)
computer. Three ideas, each with a clean intuition:

### Why you can't just "solve" a giant system directly

The textbook way to solve a linear system is **elimination** (organized
substitution — what you did by hand for two equations in two unknowns). For these
3D problems, elimination has a fatal flaw: it *fills in* the matrix with
enormous numbers of new nonzero entries as it proceeds. At 10 million unknowns the
intermediate data would be **terabytes** and take hours. Direct elimination simply
doesn't scale in 3D.

### The fix: guess-and-improve (iterative solving)

Instead of solving exactly in one expensive shot, **iterate**: start with a guess
for all 10 million unknowns and repeatedly nudge it toward the true answer, using
only cheap multiplications (no fill-in, modest memory). The method used here is
**Conjugate Gradient**, which is provably efficient *for the symmetric, stable
systems that elasticity produces*. Each iteration is cheap; the question is *how
many* iterations.

### The accelerator: multigrid, and "seeing the big picture"

Naive iteration has a notorious weakness: it fixes *local, fine-grained* error
quickly but is agonizingly slow at correcting *smooth, large-scale* error — the
overall "this whole region needs to shift" kind. Imagine smoothing a wrinkled
bedsheet: you can flatten small wrinkles by patting locally, but a big diagonal
ripple across the whole sheet needs you to *step back and grab the corners*.

**Algebraic Multigrid (AMG)** is exactly "stepping back." It automatically builds
a hierarchy of coarser and coarser versions of the problem. Fine levels fix fine
wrinkles; coarse levels fix the sheet-wide ripples cheaply; the answer is
assembled across all scales. The payoff is decisive: with good multigrid the
number of iterations stays **roughly constant no matter how big the model gets**.
That's the difference between work that grows *linearly* with the problem size
(affordable) and work that grows much faster (not).

There's a subtle but crucial detail this project had to get right. A 3D solid has
six "free rides" — motions that cost no energy: sliding it in x, y, or z, and
rotating it about each axis (a rigid object floating in space). Multigrid's
coarse approximations must *know about* these six rigid-body motions, or its
big-picture corrections are subtly wrong and the iteration count creeps up with
size. Feeding multigrid these six modes explicitly was the single change that took
the iteration count from "slowly growing" to genuinely **flat** — restoring the
linear scaling. (We measured both: without the hint, work grew as roughly
size^1.3; with it, essentially size^1.0. The test suite now guards this.)

The same "think about scale" discipline shows up in memory. The program never
stores anything that grows wastefully: it exploits the fact that a regular grid's
millions of bricks are *all identical* (store one, not millions), and it builds
the giant sparse matrix's structure directly rather than through a bloated
intermediate. The result: a 10-million-unknown plasticity model fits in about
**47 GB** — inside a 64 GB workstation — and the work to solve it grows in
proportion to the model size, which is the best one can hope for.

---

## 6. When the bending is for real: finite strain

Everything so far quietly assumed the deformations are **small** — that the shape
barely changes, so we can do the bookkeeping on the *original* geometry. That's
fine for a bridge that sags a few millimeters. It is hopelessly wrong for a
paperclip you fold in half, a metal can you crush, or a tensile bar that stretches
until it **necks** down and snaps. There the rotations are large and the geometry
changes so much that "where things are" is part of the unknown. This regime is
called **finite strain** (or large deformation), and the program handles it as a
second mode you switch on with one keyword.

Two ideas make it work.

**You can't just add up stretches anymore — you multiply them.** At small strain,
total = elastic + plastic, a simple sum. At large strain, deformation *compounds*
like compound interest: the program splits the total stretch into a plastic part
(the permanent reshaping) followed by an elastic part (the springy bit on top),
**multiplied** together rather than added. Picture stretching a rubber sheet that
already has a permanent bulge: the new stretch acts *on top of* the bulged shape,
so the order and the geometry matter.

**The beautiful trick: change your ruler and the old machinery comes back.** This
is the part that makes the whole thing tractable. If you measure stretch on a
**logarithmic** ruler (so "stretch by 2×" and "stretch by 2× again" *add up* to
"4×"), and you first mentally **un-rotate** the material to a neutral pose, then —
remarkably — the large-strain problem looks *algebraically identical* to the
small-strain one in that rotated, log-stretched frame. So the exact same,
already-verified "return mapping" from §4 is reused without change; finite strain
is a geometric *wrapper* that translates into and out of this convenient frame.
(This is a classical result — Simo, 1992.) A nice bonus falls out for free:
because plastic flow shuffles shape but doesn't change volume, the logarithmic
bookkeeping conserves volume **exactly**, which the small-strain version could only
approximate.

This buys two things you genuinely can't get otherwise:

- **Necking.** Pull a bar hard enough and it doesn't thin uniformly — at some point
  the deformation *localizes* into a narrowing neck that runs away to failure. That
  is a large-deformation, large-rotation, nearly-incompressible phenomenon; the
  program reproduces the classic necking benchmark.
- **Large rotations and buckling** — a cantilever that swings through a big angle, a
  shaft twisted hard. The forces must be computed on the *deformed* shape, which the
  finite-strain "initial-stress" stiffness accounts for.

Two subtleties are worth naming, because getting them wrong is a classic trap:

- **Objectivity (the physics can't depend on how you're looking at it).** If you
  pick up a stressed object and merely *rotate* it, no new stress should appear.
  Naively-stored internal memory (the "back-stress" of kinematic hardening) breaks
  this — it doesn't turn with the object. The fix is to store that memory in the
  un-rotated frame and rotate it back in on demand, so the model gives the same
  answer no matter how the object is oriented.
- **Locking, and the F-bar cure.** Plastic flow preserves volume, and simple brick
  elements are bad at deforming while *exactly* preserving volume — they go
  artificially stiff ("locking"). A standard remedy, **F-bar**, lets each element
  take its volume-change from a single representative point, relaxing the
  over-constraint. It's the large-strain version of a well-known small-strain trick.

One practical consequence: in these harder cases the big matrix the solver works
with is no longer **symmetric** (a property the fast Conjugate-Gradient method
relied on). The program detects this and quietly switches to a solver that doesn't
need symmetry — so the user still just calls `solve!`.

---

## 7. Why this is a satisfying piece of computational science

Plasticity sits at a sweet spot. The physics is **tangible** — you can feel it in
a paperclip — yet encoding it faithfully forces you through real depth: a yield
*surface* instead of a threshold, a material that *remembers its path*, a stress
that is *constrained* to a moving boundary. And solving it at scale forces a second
kind of depth: the realization that *how* you solve a million equations matters as
much as *what* the equations say — that elimination dies in 3D, that iteration
plus multigrid plus "remember the rigid-body motions" restores tractability, and
that careful bookkeeping is the difference between fitting in memory and not.

The program in this repository is a compact, tested embodiment of both: a faithful
miniature of how metal yields, and an efficient machine for predicting it on
shapes and at sizes no one could work out by hand.

---

### A one-paragraph version

*Metal springs back under small loads but takes a permanent set under large ones;
the boundary is a "yield surface" the internal stress isn't allowed to cross, and
the material flows just enough to stay on it — remembering its whole loading
history as it goes. To compute this for a real 3D shape, we chop the shape into
millions of little bricks, track how their corners move, and replay the load in
small steps, at each step using Newton's method to find the balanced shape and a
"return mapping" to enforce the yield surface point-by-point. The resulting
10-million-equation systems are too big to solve by direct elimination, so we
solve them by guess-and-improve iteration accelerated by multigrid — which, once
told about the six rigid-body motions of a free solid, converges in a near-constant
number of steps, making the whole thing scale linearly and fit on an ordinary
workstation.*
