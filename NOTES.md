# Project notes

## Naming

Originally I used terms "candidate" and "citizen/voter" fairly consistently.
This tool is not primary about simulating political elections, thus "citizen" is
not an ideal name. Voter is preferred, and is now used throughout (including
`nvtr`/`ivtr` for counts and indices).

"Candidate" is also not great. It implies that the options that voters express
their opinions on are people. "Options", "Choices", or "Alternatives" are
better. "Alternatives" is too long, so... In programming and user interfaces,
config etc., the term "option" is more overloaded. So I'm going to use
"choice(s)" throughout, gradually.

## To-Do

* Cargo clippy needs to be happy
* Apply "no raw loops" principles as much as I can
* Rename candidate -> choice (citizen -> voter is done)
* Clean up Issues
  * Horizon feels problematic -- no slope beyond it. Substitute with Gaussian as
    an option. Does this imply re-structuring? Probably yes, because of the way
    multiple dimensions add.
* New consideration, IssueFactions
* Strategic voting improvements
  * Allow a fraction of the population to be strategic
  * Allow a political faction to be more strategic than another
    * Under different methods, how much are strategic voters wrongly rewarded?
  * Each voting method needs a new method, strategic_prereq to return
    Option<Method>.
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


## Coverage

```bash
cargo watch -d 2 -x 'llvm-cov nextest --lcov --output-path=./target/lcov.info' -x 'llvm-cov report'
```
