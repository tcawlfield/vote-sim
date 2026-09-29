# Project notes

## Naming

Originally I used terms "candidate" and "citizen/voter" fairly consistently.
This tool is not primary about simulating political elections, thus "citizen" is
not an ideal name. Voter is preferred, and is now used throughout (including
`nvtr`/`ivtr` for counts and indices).

"Candidate" is used consistently here but also in voting method and simulation
literature. "Alternative" is also used when it's desirable to distinguish
between voting on people rather than options in general. But "alternative" feels
a little strained in the context of this code.

## To-Do

* Strategic voting improvements
  * Should I expand the runner-up concept? How?
  * Allow a fraction of the population to be strategic
  * Allow a political faction to be more strategic than another
    * Under different methods, how much are strategic voters wrongly rewarded?
* Add a Virtues (described above) consideration
* Multi-winner methods
  * Iterative rewreighted range voting ---- loop through winners, removing them
    and adding another in their place. Keep cycling through winners until either
    the winner list becomes stable (needs better definition) or a maximum cycle
    count is reached. Research prior art here, and consider repeating stability
    patterns.
    * This needs to be measured against a computationally-expensive method that
      picks the best set out of all combinations of winning committee.
  * How should we best characterize the effectiveness of the winning set?
    * For each candidate, score utility by winners in preference order. Most
      preferred gets 100%, next-most gets ... 50% maybe? Etc. What is natural
      here?


## Coverage

```bash
cargo watch -d 2 -x 'llvm-cov nextest --lcov --output-path=./target/lcov.info' -x 'llvm-cov report'
```
