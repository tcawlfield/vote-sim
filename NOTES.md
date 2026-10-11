# Project notes

## To-Do

* Strategic voting improvements
  * Should I expand the runner-up concept? How?
  * Allow a fraction of the population to be strategic
  * Allow a political faction to be more strategic than another
    * Under different methods, how much are strategic voters wrongly rewarded?
* Add a Virtues (described above) consideration
* Multi-winner methods
  * Iterative reweighted range voting ---- loop through winners, removing them
    and adding another in their place. Keep cycling through winners until either
    the winner list becomes stable (needs better definition) or a maximum cycle
    count is reached. Research prior art here, and consider repeating stability
    patterns.

## Coverage

```bash
cargo watch -d 2 -x 'llvm-cov nextest --lcov --output-path=./target/lcov.info' -x 'llvm-cov report'
```
