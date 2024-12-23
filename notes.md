# Notes

## 2024-12-05

sometimes the AO will backtrack over a space it has already been to, but the population for that point in the path will have already been updated, I need to figure out what to do about that

I think because it is reinforcing the backtracking, the AO is actually getting stuck in a loop where it just walks back and forth between two points, I need to figure out how to stop that

- This issue is somewhat resolved by reinforcement decay; however, the AO still sometimes gets stuck in a loop especially after many trials

## 2024-12-10

Right now the algorithm only works when each state is only updated one time. I am basically looking at the most recent time a behavior was emitted from that state and reinforcing it. I'm not totally sure this can be justified. The reason that not doing this causes issues is because the AO will be reinforced for a behavior the first time then the population will shift to reflect that behavior; however, if the population gets reinforced again it will be reinforced for a behavior that is no longer in the new population which causes issues for a linear FDF if no parents are eligible. I think that using an exponential FDF might solve this problem.
