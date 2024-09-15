# My Voting Simulation

Here is (yet another) voting simulation.

## Motivation

## Why program this in Rust?

It's helpful to have a fast simulation even when there are many voters, candidates,
and methods. Also, it's helpful for me to practice and better understand Rust.

## Original question: Does strategic voting do better or worse for plurality?

A very early voting simulation (TODO: Reference?) apparently had the result that strategic voters,
as a population, always performed worse for themselves (collectively) than
honest voters. But for the *plurality* voting method, that seems like an irrational
result, and [other voting simulations](https://electionscience.github.io/vse-sim/)
gave opposite results. I wanted to weigh in on this, and my simulation here produces
the result that in *plurality* voting, a population of strategic voters do better for themselves.

For all other methods considered here, strategic voters tend to get worse results for the
whole population.

Consider STAR voting. STAR uses a score-based ballot. The suggested scale is 0-5, or six
possible scores for each candidate. Step one: Add up the scores for each candidate. Find the
*two* highest-scoring candidates. Step two: do an (instant) runoff between these, using
every ballot where these two finalists were given different scores. This runoff is
a plurality vote, counting the numbers of ballots that favor each one candidate over the other.
* Using plurality to pick the final winner helps minimize strategic effects in two ways:
  * Degree of score-separation no longer is considered
  * Voters are encouraged to differentiate between candidates, especially any who are
    "hopefuls" (expected potential winners). This may discourage ballots with all zeros
    and fives.
* **This is somewhat similar to strategic voting with plurality!** It's still better than
  plurality because the top-two are chosen with a method that is more expressive, and
  has neither spoiler nor center-squeeze effects. But it ends with plurality, which
  for 3+ candidates is a terrible method by all measures, but for exactly two
  candidates is probably the very best method available.

So by reason of that, plurality voting with strategic voters is much less horrible than it
is with honest voters.

## Thoughts

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

### Sortition

This project is all about voting, a topic that goes beyond politics. But it interfaces
with politics. So I allow myself a small space to opine about sortition here. Sortition is
the idea of electing people at random, by lottery.

It is clear that voters have more accurate and incisive opinions of candidates that they
know well personally than people they do not know. In anything other than small-town, local
elections, voters have very little first-hand knowledge of the candidates. It follows that
the huge majority of voters do not contribute any real or authentic information about
the candidates by way of their ballots. They only contribute information about their
perceived values or needs. This information is filtered through the lens of what voters
believe that the candidates themselves stand for or support, and how their campaigns
have shaped the minds of the electorate and have manufactured (usually) dislike and even
outrage over the supposed policies and personalities of other candidates. One way or
another it's a battle of perception, not reality.

It seems that this leads to a lot of problems that are manifest in politics today.
Political parties consolidate power, and thus silence individual voices and
weaken the ability for any legislative
assemblies to make decisions that favor the common good. Effective leadership requires
finding solutions that meet the needs of the represented as completely as possible.
"I win, you win" kinds of solutions. But partisan politics encourage adversarial
decision-making, resulting in policies designed to favor certain groups at the
expense of others.

If we wish to have both accurate representation as well as representatives who are encouraged
to promote general welfare, then we may want to select citizens for public service
who would not be inclined to serve if it required a popularity contest or sacrificing
principles for campaign funds and the support of one political party or another.
The very nature of elections is competition. Perhaps large public elections find the
worst of us, not the best, regardless of the merits of any voting system.

## Code

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

## To-Do

* Fix strategic ballot scores
* Allow a fraction of the population to be strategic
* Allow a political faction to be more strategic than another
  * Under different methods, how much are strategic voters wrongly rewarded?

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
