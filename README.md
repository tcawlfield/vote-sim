# SimVote

Simulate some voting methods.
* Does strategic voting do better or worse for plurality? Even when issue-space is 1-D?
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
