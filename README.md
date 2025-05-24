# My Voting Simulation

Here is (yet another) voting simulation.

## Motivation

## Why program this in Rust?

It's helpful to have a fast simulation even when there are many voters, candidates,
and methods. Also, it's helpful for me to practice and better understand Rust.

## Analysis

The simulation is controlled by a configuration file, and writes output to a Parquet file.
You can study the results of these simulations using Python. I generally do this with
Jupyter notebooks and a variety of libraries including the excellent Awkward Array
project.

## Blog

Here are some results from this simulation:

[How we all Chose](https://tcawlfield.github.io/vote-sim-studies/)

## Code

Note: I would like to get Rust documentation up on Github Pages. Not done yet though.

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

### Perception versus reality

First thought: A voting method attempts to capture the collective will of the people.
It is not the role of a voting method to address the difference between perceived utility
and actual utility that a canditate may provide to a voter.

Second thought: It has been observed that some collective activities
result in "collective intelligence." But other activities can produce collective stupidity.
A good voting method ought to result in collective intelligence.
So it may be possible for a voting method to help maximize *actual utility* even while
individual ballots can only be based on percieved utility. This should only be possible
if perceived utility, on average, has a positive correlation with actual utility.
This can be modelled in a simulation, provided there is some distinction made
between (simulated) actual utility and perceived, which would be affected by biases.

### Ideal virtues of voting methods

* Net utility of the winning candidate should be close to that of the ideal candidate
  * Compare average VSE, worst-case, and fraction of elections that yield the ideal
* Ballots are expressive
* It is obvious and natural how to best fill out your ballot
  * It should not be tedious or require too much specificity
* Various criteria should be satisfied more often than not
* The method should be relatively easy to explain
* Honest voting should be nearly as effective as any kind of strategic voting
* Votes can be tabulated within precincts in a generally compact way.
  * For each method, this compactness is a function of the number of candidates
    and, for score-based systems, the number of scale degrees.
  * This can help for auditing purposes

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

### More questions:

* How does dimensionality of issue space affect things?
* Can we generate candidates in a way that mimics real election results?
  * Try generating candidates in position-space by doing a multi-seat pre-election. This
    might help distribute candidates more uniformly in position space.
* How does candidate generation affect regret?
* Resistance against manipulation
  * If one point in issue space correlates with (encourages) strategic voting, does this influence election results?
  * Does it reduce manipulation to rescale approval scores?
* Compare range voting with plurality, approval, and IRV/RCV.
  * Debate exists about ranked choice versus Bayesian regret
  * Approval voting avoids degrees of favor, but conveys less information than range & IRV.
    How does it stack up against IRV in particular? Are there alternative evaluation systems to
    regret?
