Require Import Arith.
Require Import Nat.
Require Import Lia.

Theorem foo:
    forall n: nat, 1 + n > n.
Proof.
Admitted.

Theorem foofoo:
    forall n: nat, 1 + n + 1 > n.
Proof.
Admitted.