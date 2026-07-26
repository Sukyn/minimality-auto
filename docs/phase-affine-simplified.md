# Single-multiplier phase-affine presentations

For $d=5$, choose $g=2$, so $M_{2^k}=M_g^k$ and $M_g^4=I$. The relation
$X^5=I$ retains the old $a=1$ instance of rule (3). For rule (5), put
$q=I\otimes M_g$ and $u=CX$. Its order and $g$-instance are replaced by
the single relation $q^2uquq=u$. To see why cancellation is valid, write
$D=SWAP;u;SWAP$: A7 makes $D^4uD^4$ invertible, hence $D$ and $u$ are
invertible. The four $q$-conjugates of the new relation then imply $u^5=I$
and $q^{-1}uq=u^2$. Conversely, those two equations imply $q^2uquq=u$.

The control multiplier law and the translation-through-$CX$ law are likewise
replaced by one affine law. With $p=M_g\otimes I$, $x=X\otimes I$ and
$y=I\otimes X$, it is $pxu=u^2pxy$. The old two rules give it directly.
Conversely, the target multiplier law and A7 imply $p^2up^2=u^{-1}$; a target
$M_g$-conjugate of the new law first gives $uy=yu$, after which its powers
recover $xux^{-1}=uy$ and $pup^{-1}=u^2$.

From the $g$-instances, induction gives
$X M_g^k=M_g^kX^{2^k}$, $M_g^k CX=CX^{2^k}M_g^k$, and
$CX M_g^k=M_g^kCX^{2^k}$, recovering every former multiplier instance
after reducing exponents modulo $5$; in the last two formulas the multiplier
is on the control and target wire, respectively. The phase multiplier
families follow by the same iteration.

For $d=7$, choose $g=3$, so $M_g^6=I$. Here the two merged affine rules are
\[
q^3uquq^2=u,
\qquad
pxu=u^3pxy.
\]
For the first, the six $q$-conjugates give $u^7=I$ and $q^{-1}uq=u^3$.
For the second, A7 gives $p^3up^3=u^{-1}$; conjugating by powers of $q$
first gives $uy=yu$, after which the same power argument recovers the two
original control rules. All multiplier families then follow from
$M_{3^k}=M_g^k$.

For $d=11$, choose $g=2$. The control-side merge remains valid, giving
$pxu=u^2pxy$, but the order and target-conjugation laws are kept separately:

\[
u^{11}=I,
\qquad
uq\,u^9=q.
\]

The tempting relation $q^7uquq^2=u$ is true in the intended interpretation,
but its conjugates do not suffice to recover these two laws: without a further
argument they present an infinite cyclically presented group. Thus the $d=11$
file deliberately has eight affine rules rather than copying the seven-rule
$d=5,7$ presentation.

Several separators are uniform for every odd prime. In particular, the exact
affine lift for the merged control rule may translate $CX$ on
$(a,0),(a,a),(-a,0),(-a,-a)$ for any nonzero $a$. Presence separates A1 and
A7; counting $X$ modulo $d+g-1$ separates A3_order; an affine lift separates
A3_g; and the exact block-normal-form argument separates A8. The two separate
A5 laws in dimension 11 remain the obstacle to turning these examples alone
into a minimality theorem for every prime.
