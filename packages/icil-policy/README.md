# icil-policy

The policy protocol of the RoboTensor one-demonstration in-context imitation learning (ICIL)
competition. A benchmark drives a policy it cannot import: the policy is served in its own process,
and the two exchange named numpy arrays and JSON fields over an authenticated socket. Nothing is
ever pickled.

It depends on numpy and PyYAML only, because it is installed into every competitor's image.

**Status:** scaffold.
