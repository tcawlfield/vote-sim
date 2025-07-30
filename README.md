# My Voting Simulation

Here is (yet another) voting simulation.

## Design

This simulation is driven by a config file, in either YAML or TOML. (JSON
support should be trival to add and might provide easier interop. So that's a
TO-DO.) It outputs a rich set of information to a Parquet file.

This is written in Rust because Rust can be fast, has rich library support, a
great unit test framework builtin, and I needed an excuse to practice Rust at
the time I started it.

## Analysis

YAML/TOML and Parquet are language-agnostic, so it is easy to do higher-level analysis
in another language. Python has been my language of choice for that, but that's
outside the immediate concern of this simulation. I generally do analysis with
Jupyter notebooks and a variety of libraries including the excellent Awkward Array
project. Some example notebooks can be found in this repository, but more can be
found at [vote-sim-studies](https://github.com/tcawlfield/vote-sim-studies).

## Blog

Here are some results from this simulation:

[How we all Chose](https://tcawlfield.github.io/vote-sim-studies/)

## Purpose

The purpose of this simulation is to make it easy to explore various aspects of
voting methods. This is not directly political. I have used this simulation as a
source of recommendations in a few small-group voting situations.

## Code

Note: Pushing to crates.io is yet another To-do.

* Useful terms:
  * Consideration / axis
    * Issue / Persuasion / Position
    * Likability
  * Score / utility
* How are scores combined?
  * At the top level we have a list of Considerations
    * Combine with Addition / Multiplication / Quadrature/distance
  * Considerations themselves can be simple or multi-dimensional
    * N-dimensional considerations could evaluate with distance or multiplication
    * I'm not convinced multiplication makes much sense. But addition vs distance is interesting.

## Concepts

This simulation uses *voter satisfaction efficiency* as a metric of success. The details
of that are not particularly important, but it assumes an abstract metric akin to
*utility* (from economics). The idea is that for each voter, each candidate has some
(1-D or scalar) utility that this candidate, if elected, would provide to that voter.
With this model, we presume that the ideal candidate would be the one which provides
the maximum utility, summed over all voters.

### Simulating utilities

We must generate a table of utilities: candidates ✕ voters.
Here are some methods to do this, which can potentially be combined (added):

#### Issue space

There exists a space of issues. Maybe one-dimensional like left versus
right, two-dimensional like the Political Compass, or more.
* Are there clusters or islands of oppinion in this space?
* Is utility a (decreasing) function of absolute distance between a voter and candidate?
  Or is the square of the distance a better representation (of perception)?
* If issue space is one-dimensional (and the only consideration), there ought to be
  no Condorcet cycles. I should test this as a cross-check of the code.

#### Likability

Each candidate has a charisma that is universally appealing.
* One dimension is enough. If there were multiple dimensions, this would be like
  Virtues, below

#### Virtues

Each candidate is characterized as expressing certain "virtues" to various degrees.
Each voter is influenced or affected by some virtues more than others.

* Can these degrees of a voters' virtue-preference be negative multipliers as well as positive?
  * If this is too common, Issue space may better represent this effect
* If there's no clumpiness in voter issue-sensitivity space, is this system interesting enough?

#### Irrational

The voter-candidate utility table can be filled with random variates.
This may work better with relatively small numbers of voters. But many systems
like approval voting require a large number of voters. So there can be
N core clusters (each one random utilities for each candidate). A voter's
cluster membership is i % N. Each voter gets a relatively-small random deviation added
to the core utility scores.

This method should be more likely to create Condorcet cycles -- no Condorcet winner.

## To-Do

* Strategic voting improvements
  * Allow a fraction of the population to be strategic
  * Allow a political faction to be more strategic than another
    * Under different methods, how much are strategic voters wrongly rewarded?
  * Each voting method needs a new method, strategic_prereq to return Option<Method>.
    * We can use this to ensure that each strategic method is preceded by its
      honest "pre-election poll" method. If not, they can be inserted.
    * At the same time, this suggests another property of MethodSim: is_visible.
      is_visible() returns false if the method was inserted as a pre-poll.
* Add a Virtues (described above) consideration
* Multi-winner methods
  * Iterative rewreighted range voting ---- loop through winners, removing them
    and adding another in their place. Keep cycling through winners until either
    the winner list becomes stable (needs better definition) or a maximum cycle
    count is reached. Research prior art here, and consider repeating stability
    patterns.
  * How should we best characterize the effectiveness of the winning set?
    * For each candidate, score utility by winners in preference order. Most
      preferred gets 100%, next-most gets ... 50% maybe? Etc. What is natural
      here?
