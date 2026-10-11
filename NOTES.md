# Project notes

## To-Do

* Strategic voting improvements
  * Should I expand the runner-up concept? How?
  * Allow a fraction of the population to be strategic.
  * Allow a political faction to be more strategic than another.
    * Under different methods, how much are strategic voters wrongly rewarded?
    * Should Electorate and Irrational collaborate on their factions? These
      are fairly different in how they handle factions. Specifically, irrational
      currently does not have user-defined popularities.
* Add a Virtues consideration.
  * Intent is to uncork the idea of perceived versus actual utility.
    (Normally this distinction is understood to be beyond the scope of voting.)
    Do different voting methods differ in terms of realizing collective intelligence
    or collective stupidity?
  * Each voter or maybe voter faction has a vector of weights for how each virtue
    affects their perceived utility. Maybe there needs to be weights for "true"
    utility? IDK. Can voter-utility weights go negative? Maybe so but averages
    across all voters must be positive if there's to be any hope of collective
    intelligence. Or maybe that virtue is really a curse, therefore a negative
    true-utility weight.
* Multi-winner methods:
  * Iterative reweighted range voting ---- loop through winners, removing them
    and adding another in their place. Keep cycling through winners until either
    the winner list becomes stable (needs better definition) or a maximum cycle
    count is reached. Research prior art here, and consider repeating stability
    patterns.
  * Existing methods such as ranked-pairs that can be continued naturally for committee
    elections. The head-scratcher here is how this may synergize with strategic
    voting info, supplying (fully?) ranked candidates instead of just `WinnerAndRunnerup`.
* Explicit (conditional) output of the voter-candidate utility matrix and actual
  ballots. This may help to generate examples of intriguing elections.
  * Follow-up, compact ballot info to report counts of unique ballots. That would
    break any correspondence with the utility matrix. Yep nope. Handle this at
    the analysis stage instead.

## Coverage

```bash
cargo watch -d 2 -x 'llvm-cov nextest --lcov --output-path=./target/lcov.info' -x 'llvm-cov report'
```
