// Some definitions presupposed by pandoc's typst output.
#let blockquote(body) = [
  #set text( size: 0.92em )
  #block(inset: (left: 1.5em, top: 0.2em, bottom: 0.2em))[#body]
]

#let horizontalrule = line(start: (25%,0%), end: (75%,0%))

#let endnote(num, contents) = [
  #stack(dir: ltr, spacing: 3pt, super[#num], contents)
]

#show terms: it => {
  it.children
    .map(child => [
      #strong[#child.term]
      #block(inset: (left: 1.5em, top: -0.4em))[#child.description]
      ])
    .join()
}

// Some quarto-specific definitions.

#show raw.where(block: true): set block(
    fill: luma(230),
    width: 100%,
    inset: 8pt,
    radius: 2pt
  )

#let block_with_new_content(old_block, new_content) = {
  let d = (:)
  let fields = old_block.fields()
  fields.remove("body")
  if fields.at("below", default: none) != none {
    // TODO: this is a hack because below is a "synthesized element"
    // according to the experts in the typst discord...
    fields.below = fields.below.abs
  }
  return block.with(..fields)(new_content)
}

#let empty(v) = {
  if type(v) == str {
    // two dollar signs here because we're technically inside
    // a Pandoc template :grimace:
    v.matches(regex("^\\s*$")).at(0, default: none) != none
  } else if type(v) == content {
    if v.at("text", default: none) != none {
      return empty(v.text)
    }
    for child in v.at("children", default: ()) {
      if not empty(child) {
        return false
      }
    }
    return true
  }

}

// Subfloats
// This is a technique that we adapted from https://github.com/tingerrr/subpar/
#let quartosubfloatcounter = counter("quartosubfloatcounter")

#let quarto_super(
  kind: str,
  caption: none,
  label: none,
  supplement: str,
  position: none,
  subrefnumbering: "1a",
  subcapnumbering: "(a)",
  body,
) = {
  context {
    let figcounter = counter(figure.where(kind: kind))
    let n-super = figcounter.get().first() + 1
    set figure.caption(position: position)
    [#figure(
      kind: kind,
      supplement: supplement,
      caption: caption,
      {
        show figure.where(kind: kind): set figure(numbering: _ => numbering(subrefnumbering, n-super, quartosubfloatcounter.get().first() + 1))
        show figure.where(kind: kind): set figure.caption(position: position)

        show figure: it => {
          let num = numbering(subcapnumbering, n-super, quartosubfloatcounter.get().first() + 1)
          show figure.caption: it => {
            num.slice(2) // I don't understand why the numbering contains output that it really shouldn't, but this fixes it shrug?
            [ ]
            it.body
          }

          quartosubfloatcounter.step()
          it
          counter(figure.where(kind: it.kind)).update(n => n - 1)
        }

        quartosubfloatcounter.update(0)
        body
      }
    )#label]
  }
}

// callout rendering
// this is a figure show rule because callouts are crossreferenceable
#show figure: it => {
  if type(it.kind) != str {
    return it
  }
  let kind_match = it.kind.matches(regex("^quarto-callout-(.*)")).at(0, default: none)
  if kind_match == none {
    return it
  }
  let kind = kind_match.captures.at(0, default: "other")
  kind = upper(kind.first()) + kind.slice(1)
  // now we pull apart the callout and reassemble it with the crossref name and counter

  // when we cleanup pandoc's emitted code to avoid spaces this will have to change
  let old_callout = it.body.children.at(1).body.children.at(1)
  let old_title_block = old_callout.body.children.at(0)
  let old_title = old_title_block.body.body.children.at(2)

  // TODO use custom separator if available
  let new_title = if empty(old_title) {
    [#kind #it.counter.display()]
  } else {
    [#kind #it.counter.display(): #old_title]
  }

  let new_title_block = block_with_new_content(
    old_title_block, 
    block_with_new_content(
      old_title_block.body, 
      old_title_block.body.body.children.at(0) +
      old_title_block.body.body.children.at(1) +
      new_title))

  block_with_new_content(old_callout,
    block(below: 0pt, new_title_block) +
    old_callout.body.children.at(1))
}

// 2023-10-09: #fa-icon("fa-info") is not working, so we'll eval "#"❕"" instead
#let callout(body: [], title: "Callout", background_color: rgb("#dddddd"), icon: none, icon_color: black, body_background_color: white) = {
  block(
    breakable: false, 
    fill: background_color, 
    stroke: (paint: icon_color, thickness: 0.5pt, cap: "round"), 
    width: 100%, 
    radius: 2pt,
    block(
      inset: 1pt,
      width: 100%, 
      below: 0pt, 
      block(
        fill: background_color, 
        width: 100%, 
        inset: 8pt)[#text(icon_color, weight: 900)[#icon] #title]) +
      if(body != []){
        block(
          inset: 1pt, 
          width: 100%, 
          block(fill: body_background_color, width: 100%, inset: 8pt, body))
      }
    )
}



#let article(
  title: none,
  subtitle: none,
  authors: none,
  date: none,
  abstract: none,
  abstract-title: none,
  cols: 1,
  margin: (x: 1.25in, y: 1.25in),
  paper: "us-letter",
  lang: "en",
  region: "US",
  font: "libertinus serif",
  fontsize: 11pt,
  title-size: 1.5em,
  subtitle-size: 1.25em,
  heading-family: "libertinus serif",
  heading-weight: "bold",
  heading-style: "normal",
  heading-color: black,
  heading-line-height: 0.65em,
  sectionnumbering: none,
  pagenumbering: "1",
  toc: false,
  toc_title: none,
  toc_depth: none,
  toc_indent: 1.5em,
  doc,
) = {
  set page(
    paper: paper,
    margin: margin,
    numbering: pagenumbering,
  )
  set par(justify: true)
  set text(lang: lang,
           region: region,
           font: font,
           size: fontsize)
  set heading(numbering: sectionnumbering)
  if title != none {
    align(center)[#block(inset: 2em)[
      #set par(leading: heading-line-height)
      #if (heading-family != none or heading-weight != "bold" or heading-style != "normal"
           or heading-color != black or heading-decoration == "underline"
           or heading-background-color != none) {
        set text(font: heading-family, weight: heading-weight, style: heading-style, fill: heading-color)
        text(size: title-size)[#title]
        if subtitle != none {
          parbreak()
          text(size: subtitle-size)[#subtitle]
        }
      } else {
        text(weight: "bold", size: title-size)[#title]
        if subtitle != none {
          parbreak()
          text(weight: "bold", size: subtitle-size)[#subtitle]
        }
      }
    ]]
  }

  if authors != none {
    let count = authors.len()
    let ncols = calc.min(count, 3)
    grid(
      columns: (1fr,) * ncols,
      row-gutter: 1.5em,
      ..authors.map(author =>
          align(center)[
            #author.name \
            #author.affiliation \
            #author.email
          ]
      )
    )
  }

  if date != none {
    align(center)[#block(inset: 1em)[
      #date
    ]]
  }

  if abstract != none {
    block(inset: 2em)[
    #text(weight: "semibold")[#abstract-title] #h(1em) #abstract
    ]
  }

  if toc {
    let title = if toc_title == none {
      auto
    } else {
      toc_title
    }
    block(above: 0em, below: 2em)[
    #outline(
      title: toc_title,
      depth: toc_depth,
      indent: toc_indent
    );
    ]
  }

  if cols == 1 {
    doc
  } else {
    columns(cols, doc)
  }
}

#set table(
  inset: 6pt,
  stroke: none
)
#import "@preview/ctheorems:1.1.3": *
#show: thmrules
#let definition = thmbox("definition", "Definition", base_level: 1)
#import "@preview/fontawesome:0.5.0": *
#let theorem = thmbox("theorem", "Theorem", base_level: 1)
#let corollary = thmbox("corollary", "Corollary", base_level: 1)
#let lemma = thmbox("lemma", "Lemma", base_level: 1)
#let proposition = thmbox("proposition", "Proposition", base_level: 1)

#show: doc => article(
  title: [BlockTensorFactorization.jl: A Unified Constrained Tensor Decomposition Julia Package],
  authors: (
    ( name: [Nicholas J. E. Richardson],
      affiliation: [Department of Mathematics],
      email: [] ),
    ( name: [Noah Marusenko],
      affiliation: [Department of Computer Science],
      email: [] ),
    ( name: [Michael P. Friedlander],
      affiliation: [Departments of Mathematics and Computer Science

],
      email: [] ),
    ),
  font: ("Libertinus Serif",),
  sectionnumbering: "1.1.a.i",
  pagenumbering: "1",
  toc: true,
  toc_title: [Table of contents],
  toc_depth: 3,
  cols: 1,
  doc,
)

= Introduction
<introduction>
- Tenors are useful in many applications
- Need tools for fast and efficient decompositions

For the scientific user, it would be most useful for there to be a single piece of software that can take as input 1) any reasonable type of factorization model and 2) constraints on the individual factors, and produce a factorization. Details like what rank to select, how the constraints should be enforced, and convergence criteria should be handled automatically, but customizable to the knowledgable user. These are the core specification for BlockTensorFactorization.jl.

== Related tools
<related-tools>
- Packages within Julia
- Other languages
- Hint at why I developed this

Beyond the external usefulness already mentioned, this package offers a playground for fair comparisons of different parameters and options for performing tensor factorizations across various decomposition models. There exist packages for working with tensors in languages like Python (TensorFlow @martin_abadi_tensorflow_2015, PyTorch @ansel_pytorch_2024, and TensorLy @kossaifi_tensorly_2019), MATLAB (Tensor Toolbox @bader_tensor_2023), R (rTensor @li_rtensor_2018), and Julia (TensorKit.jl @jutho_juthotensorkitjl_2024, Tullio.jl @abbott_mcabbotttulliojl_2023, OMEinsum.jl @peter_under-peteromeinsumjl_2024, and TensorDecompositions.jl @wu_yunjhongwutensordecompositionsjl_2024). But they only provide a groundwork for basic manipulation of tensors and the most common tensor decomposition models and algorithms, and are not equipped to handle arbitrary user defined constraints and factorization models.

Some progress towards building a unified framework has been made @xu_BlockCoordinateDescent_2013@kim_algorithms_2014@yang_unified_2011. But these approaches don't operate on the high dimensional tensor data natively and rely on matricizations of the problem, or only consider nonnegative constraints. They also don't provide an all-in-one package for executing their frameworks.

== Contributions
<contributions>
- Fast and flexible tensor decomposition package
- Framework for creating and performing custom
  - tensor decompositions
  - constrained factorization (the what)
  - iterative updates (the how)
- Implement new "tricks"
  - a (Lipschitz) matrix stepsize for efficient sub-block updates
  - multi-scaled factorization when tensor entries are discretizations of a continuous function
  - partial projection and rescaling to enforce linear constraints (rather than Euclidean projection)
- ?? rank detection ??

The main contribution is a description of a fast and flexible tensor decomposition package, along with a public implementation written in Julia: BlockTensorFactorization.jl. This package provides a framework for creating and performing custom tensor decompositions. To the author's knowledge, it is the first package to provide automatic factorization to a large class of constrained tensor decompositions problems, as well as a framework for implementing new constraints and iterative algorithms. This paper also describes three new techniques not found in the literature that empirically convergence faster than traditional block-coordinate descent.

= Tensor Decompositions
<tensor-decompositions>
- the math section of the paper

This section reviews the notation used throughout the paper and commonly used tensor decompositions.

== Notation
<notation>
- tensor notation, use MATLAB notation for indexing so subscripts can be used for a sequence of tensors

=== Sets
<sets>
The set of real number is denoted as $bb(R)$ and its restrictions to nonnegative numbers is denoted as $bb(R)_(+) = bb(R)_(gt.eq 0) = {x in bb(R) mid(bar.v) x gt.eq 0}$.

We use $[N] = { 1 \, 2 \, dots.h \, N } = { n }_(n = 1)^N$ to denote integers from $1$ to $N$.

Usually, lower case symbols will be used for the running index, and the capitalized letter will be the maximum letter it runs to. This leads to the convenient shorthand $i in [I]$, $j in [J]$, etc.

We use a capital delta $Delta$ to denote sets of vectors or higher order tensors where the slices or fibres along a specified dimension sum to $1$, i.e.~generalized simplexes.

Usually, we use script letters ($cal(A) \, cal(B) \, cal(C) \,$ etc.) for other sets.

=== Vectors, Matrices, and Tensors
<vectors-matrices-and-tensors>
Vectors are denoted with lowercase letters ($x$, $y$, etc.), and matrices and higher order tensors with uppercase letters (commonly $A$, $B$, $C$ and $X$, $Y$, $Z$). The order of a tensor is the number of axes it has. We would call vectors "order-1" or "1st order" tensors, and matrices "order-2" or "2nd order" tensors.

To avoid confusion between entries of a vector/matrix/tensor and indexing a list of objects, we use square brackets to denote the former, and subscripts to denote the later. For example, the entry in the $i$th row and $j$th column of a matrix $A in bb(R)$ is $A [i \, j]$. This follows MATLAB/Julia notation where `A[i,j]` points to the entry $A [i \, j]$. We contrast this with a list of $I$ objects being denoted as $a_1 \, dots.h \, a_I$, or more compactly, ${ a_i }$ when it is clear the index $i in [I]$.

The transpose $A^tack.b in bb(R)^(J times I)$ of a matrix $A in bb(R)^(I times J)$ flips entries along the main diagonal: $A^tack.b [j \, i] = A [i \, j]$. In Julia, the transpose of a matrix is typed with a single apostrophe `A'`.

The $n$-slices, $n$th mode slices, or mode $n$ slices of an $N$th order tensor $A$ are notated with the slice $A [: \, med dots.h \, med : \, med i_n \, med : \, med dots.h \, med :]$. For a $3$rd order tensor $A$, the $1$st, $2$nd, and $3$rd mode slices $A [i \, : \, :]$, $A [: \, j \, :]$, and $A [: \, : \, k]$ have special names and are called the horizontal, lateral, and frontal slices and are displayed in #ref(<fig-tensor-slices>, supplement: [Figure]). In Julia, the 1-, 2-, and 3-slices of a third order array `A` would be `eachslice(A, dims=1)`, `eachslice(A, dims=2)`, and `eachslice(A, dims=3)`.

#figure([
#box(image("figure/tensor_slices.png"))
], caption: figure.caption(
position: bottom, 
[
Slices of an order $3$ tensor $A$.
]), 
kind: "quarto-float-fig", 
supplement: "Figure", 
)
<fig-tensor-slices>


The $n$-fibres, $n$th mode fibres, or mode $n$ fibres of an $N$th order tensor $A$ are denoted $A [i_1 \, med dots.h \, med i_(n - 1) \, med : \, med i_(n + 1) \, med dots.h \, med i_N]$. For example, the 1-fibres of a matrix $M$ are the column vectors \
$M [: \, med j]$, and the 2-fibres are the row vectors $M [i \, med :]$. For order-3 tensors, the $1$st, $2$nd, and $3$rd mode fibres $A [: \, j \, k]$, $A [i \, : \, :]$, and $A [i \, j \, :]$ are called the vertical/column, horizontal/row, and depth/tube fibres respectively and are displayed in #ref(<fig-tensor-fibres>, supplement: [Figure]). Natively in Julia, the 1-, 2-, and 3-fibres of a third order array `A` would be `eachslice(A, dims=(2,3))`, `eachslice(A, dims=(1,3))`, and `eachslice(A, dims=(1,2))`. BlockTensorFactorization.jl defines the function `eachfibre(A; n)` to do exactly this. For example, the 1-fibres of an array `A` would be `eachfibre(A, n=1)`.

For matrices, the 1-fibres are the same as the 2-slices (and vice versa), but for $N$th order tensors in general, fibres are always vectors, whereas $n$-slices are $(N - 1)$th order tensors.

#figure([
#box(image("figure/tensor_fibres.png"))
], caption: figure.caption(
position: bottom, 
[
Fibres of an order $3$ tensor $A$.
]), 
kind: "quarto-float-fig", 
supplement: "Figure", 
)
<fig-tensor-fibres>


Since we commonly use $I$ as the size of a tensor's dimension, we use $upright(i d)_I$ to denote the identity tensor of size $I$ (of the appropriate order). When the order is $2$, $upright(i d)_I$ is an $I times I$ matrix with ones along the main diagonal, and zeros elsewhere. For higher orders $N$, this is an $underbrace(I times dots.h.c times I, N upright("times"))$ tensor where $upright(i d)_I [i_1 \, dots.h \, i_N] = 1$ when $i_1 = dots.h = i_N in [I]$, and is zero otherwise.

BlockTensorFactorization.jl defines `identity_tensor(I, ndims)` to construct $upright(i d)_I$.

For a vector, matrix, or tensor filled with ones, we use $bb(1) in bb(R)^(I_1 times dots.h.c times I_N)$. This can be constructed in Julia with `ones(I₁, ..., Iₙ)`.

=== Products of Tensors
<products-of-tensors>
#definition()[
The outer product $times.circle$ between two tensors $A in bb(R)^(I_1 times dots.h.c times I_M)$ and $B in bb(R)^(J_1 times dots.h.c times J_N)$ yields an order $M + N$ tensor $A times.circle B in bb(R)^(I_1 times dots.h.c times I_M times J_1 times dots.h.c times J_N)$ that is entry-wise

$ (A times.circle B) [i_1 \, dots.h \, i_M \, j_1 \, dots.h \, j_N] = A [i_1 \, dots.h \, i_M] B [j_1 \, dots.h \, j_N] . $

] <def-outer-product>
TODO Define in BlockTensorFactorization.jl

The Frobenius inner product between two tensors $A \, B in bb(R)^(I_1 times dots.h.c times I_N)$ yields a real number $A dot.op B in bb(R)$ and is defined as

$ ⟨A \, B⟩ = A dot.op B = sum_(i_1 = 1)^(I_1) dots.h sum_(i_N = 1)^(I_N) A [i_1 \, dots.h \, i_N] B [i_1 \, dots.h \, i_N] . $

Julia's standard library package LinearAlgebra implements the Frobenius inner product with `dot(A, B)` or `A ⋅ B`.

The $n$-slice dot product $dot.op_n$ between two tensors $A in bb(R)^(K_1 \, dots.h \, K_(n - 1) \, I \, K_(n + 1) \, dots.h \, K_N)$ and $B in bb(R)^(K_1 \, dots.h \, K_(n - 1) \, J \, K_(n + 1) \, dots.h \, K_N)$ returns a matrix $(A dot.op_n B) in bb(R)^(I times J)$ with entries

$ (A dot.op_n B) [i \, j] = sum_(k_1 dots.h k_(n - 1) k_(n + 1) dots.h k_N) A [k_1 \, dots.h \, k_(n - 1) \, i \, k_(n + 1) \, dots.h \, k_N] B [k_1 \, dots.h \, k_(n - 1) \, j \, k_(n + 1) \, dots.h \, k_N] . $

This product can also be thought of as taking the dot product $(A dot.op_n B) [i \, j] = A_i dot.op B_j$ between all pairs of $n$th order slices of $A$ and $B$, which exactly how BlockTensorFactorization.jl defines the operation.

```julia
function slicewise_dot(A::AbstractArray, B::AbstractArray; dims=1)
    C = zeros(size(A, dims), size(B, dims))
    if A === B # use faster routine if they are the same
        return _slicewise_self_dot!(C, A; dims)
    end

    for (i, A_slice) in enumerate(eachslice(A; dims))
        for (j, B_slice) in enumerate(eachslice(B; dims))
            C[i, j] = A_slice ⋅ B_slice
        end
    end
    return C
end

function _slicewise_self_dot!(C, A; dims=1)
    enumerated_A_slices = enumerate(eachslice(A; dims))
    for (i, Ai_slice) in enumerated_A_slices
        for (j, Aj_slice) in enumerated_A_slices
            if i > j
                continue
            else # only compute the upper triangle entries of C
                C[i, j] = Ai_slice ⋅ Aj_slice
            end
        end
    end
    return Symmetric(C) # indexing C[2,1] points to the entry in C[1,2]
end
```

BlockTensorFactorization.jl defines this operation with `slicewise_dot(A, B, n)`. In the special case where $A = B$, a more efficient method that only computes entries where $i lt.eq j$ is defined since $A dot.op_n A$ is a symmetric matrix.

The $n$-slice product of a tensor with itself $X dot.op_n X$ should be thought of as a generalization of the Gram matrix $X^tack.b X$ since it considers the matrix generated by taking the dot product between every $n$th mode slice, just like how the Gram matrix considers the dot product between every pair of columns.

The $n$-mode product $times_n$ between a tensor $A in bb(R)^(I_1 times dots.h.c times I_N)$ and matrix $B in bb(R)^(I_n times J)$, returns a tensor $(A times_n B) in bb(R)^(I_1 times dots.h.c times I_(n - 1) times J times I_(n + 1) times dots.h.c times I_N)$ with entries

$ (A times_n B) [i_1 \, dots.h \, i_(n - 1) \, j \, i_(n + 1) \, dots.h \, i_N] = sum_(i_n = 1)^(I_n) A [i_1 \, dots.h \, i_(n - 1) \, i_n \, i_(n + 1) \, dots.h \, i_N] B [i_n \, j] . $

BlockTensorFactorization.jl defines this operation with `nmode_product(A, B, n)`.

```julia
function nmode_product(A::AbstractArray, B::AbstractMatrix, n::Integer)
    # convert the problem to the mode-1 product
    Aperm = swapdims(A, n)
    Cperm = Aperm ×₁ B
    return swapdims(Cperm, n) # swap back
end

function ×₁(A::AbstractArray, B::AbstractMatrix)
    # Turn the 1-mode product into matrix-matrix multiplication
    sizeA = size(A)
    Amat = reshape(A, sizeA[1], :)

    # Initialize the output tensor
    C = zeros(size(B, 1), sizeA[2:end]...)
    Cmat = reshape(C, size(B, 1), prod(sizeA[2:end]))

    # Perform matrix-matrix multiplication Cmat = B*Amat
    mul!(Cmat, B, Amat)

    return C # Output entries of Cmat in tensor form
end

function swapdims(A::AbstractArray, a::Integer, b::Integer=1)
    # Construct a permutation where a and b are swapped
    # e.g. [4, 2, 3, 1, 5, 6] when a=4 and b=1
    dims = collect(1:ndims(A))
    dims[a] = b; dims[b] = a
    return permutedims(A, dims)
end
```

#block[
#callout(
body: 
[
If we were only working with a fixed order of tensors, we could have defined `×₁` entry-wise with `Tullio.jl`. The function definition `tullio×₁` below gives an example for order three tensors.

```julia
function tullio×₁(A::AbstractArray{_,3}, B::AbstractMatrix)
  @tullio C[i, j, k] := A[r, j, k] * B[i, r]
  return C
end
```

But we would need a new definition for each ordered tensor, or use Julia's meta programming to write a method for each order at runtime.

]
, 
title: 
[
Note
]
, 
background_color: 
rgb("#dae6fb")
, 
icon_color: 
rgb("#0758E5")
, 
icon: 
"❕"
, 
body_background_color: 
white
)
]
The $n$-mode product and $n$-slice product can be thought of as opposites of each other. The $n$-mode product sums over just the $n$th dimension of the first tensor, whereas the $n$-slice product sums over all but the $n$th dimension.

We can extend the $n$-mode product to sum over multiple indices between two tensors.

The multi-mode product $times_(1 \, dots.h \, n) = times_(1 : n) = times_([n])$ between a tensor $A in bb(R)^(I_1 times dots.h.c times I_N)$ and tensor $B in bb(R)^(I_1 times dots.h.c times I_n)$, returns a tensor $(A times_([n]) B) in bb(R)^(I_(n + 1) times dots.h.c times I_N)$ with entries

$ (A times_([n]) B) [i_(n + 1) \, dots.h \, i_N] = sum_(i_1 \, dots.h \, i_n) A [i_1 \, dots.h \, i_n \, i_(n + 1) \, dots.h \, i_N] B [i_1 \, dots.h \, i_n] . $

This product contracts the first $n$ indexes of $A$ with every index of $B$.

More generally, we can contract any number of indexes such as the last $n$ indexes of $A$ with every index of $B$ with $times_(N - n + 1 \, dots.h \, N) = times_((N - n + 1) : N) = times_([- n])$,

$ (A times_([- n]) B) [i_1 \, dots.h \, i_n] = sum_(i_(n + 1) \, dots.h \, i_N) A [i_1 \, dots.h \, i_(N - n) \, i_(N - n + 1) \, dots.h \, i_N] B [i_(N - n + 1) \, dots.h \, i_N] \, $

or specific indexes. For example, we would define $(A times_(1 \, 3 \, 5) B) in bb(R)^(I_2 times I_4 times I_6)$ where $A in bb(R)^(I_1 times dots.h.c times I_6)$ and $B in bb(R)^(I_1 times I_3 times I_5)$ to be

$ (A times_(1 \, 3 \, 5) B) [i_2 \, i_4 \, i_6] = sum_(i_1 \, i_3 \, i_5) A [i_1 \, i_2 \, i_3 \, i_4 \, i_5 \, i_6] B [i_1 \, i_3 \, i_5] . $

When $A$ a #emph[half-symmetric];#footnote[For example, the Hessian of a scalar function (see #ref(<def-hessian>, supplement: [Definition])).] tensor of order $2 N$

#math.equation(block: true, numbering: "(1)", [ $ A [i_1 \, dots.h \, i_N \, i_(N + 1) \, dots.h \, i_(2 N)] = A [i_(N + 1) \, dots.h \, i_(2 N) \, i_1 \, dots.h \, i_N] \, $ ])<eq-half-symmetric>

we have

$ A times_([N]) B = A times_([- N]) B $

for tensors $B$ of order $N$.

=== Gradients, Norms, and Lipschitz Constants
<gradients-norms-and-lipschitz-constants>
#definition("Gradient")[
The gradient $nabla f : bb(R)^(I_1 times dots.h.c times I_N) arrow.r bb(R)^(I_1 times dots.h.c times I_N)$ of a (differentiable) function $f : bb(R)^(I_1 times dots.h.c times I_N) arrow.r bb(R)$ is defined entry-wise for a tensor $A in bb(R)^(I_1 times dots.h.c times I_N)$ by

$ nabla f (A) [i_1 \, dots.h \, i_N] = frac(partial f, partial A [i_1 \, dots.h \, i_N]) (A) . $

] <def-gradient>
#definition("Hessian")[
The Hessian $nabla^2 f : bb(R)^(I_1 times dots.h.c times I_N) arrow.r bb(R)^((I_1 times dots.h.c times I_N)^2)$ of a second-differentiable function $f : bb(R)^(I_1 times dots.h.c times I_N) arrow.r bb(R)$ is the gradient of the gradient and is defined for a tensor $A in bb(R)^(I_1 times dots.h.c times I_N)$ entry-wise by

$ nabla^2 f (A) [i_1 \, dots.h \, i_N \, j_1 \, dots.h \, j_N] = frac(partial^2 f, partial A [i_1 \, dots.h \, i_N] partial A [j_1 \, dots.h \, j_N]) (A) . $

] <def-hessian>
For a tensor input $A$ of order $N$, the Hessian tensor $nabla^2 f (A)$ is of order $2 N$. This contrasts the Hessian #emph[matrix] constructed by vectorizing the function $f$'s input @magnus_matrix_2019[Ch. 10].

See #ref(<sec-hessian-from-gradient>, supplement: [Section]) for how this definition can be reproduced by performing two gradients $nabla^2 f = nabla (nabla f)$.

#definition()[
The Frobenius norm of a tensor $A$ is the square root of its dot product with itself

$ norm(A)_F = sqrt(⟨A \, A⟩) . $

] <def-frobenius-norm>
For vectors $v$, this is equivalent to the (Euclidean) 2-norm

$ norm(v)_F = norm(v)_2 = sqrt(⟨v \, v⟩) . $

For matrices $M$, the ($2 arrow.r 2$) operator norm is defined as

$ norm(M)_(upright("op")) = sup_(lr(bar.v.double v bar.v.double)_2 = 1) norm(M v)_2 = sigma_1 (M) $

where $sigma_1 (M)$ is the largest singular value of $M$.

For tensors $T$, the operator-norm is ambiguous since there are multiple ways we can treat tensors as function on other tensors. There is a canonical way to do this for vectors $x arrow.r.bar v^tack.b x$ and matrices $x arrow.r.bar M x$, but not tensors. In the case of the Hessian tensor $nabla^2 f (A) in bb(R)^((I_1 times dots.h.c times I_N)^2)$ evaluated at $A in bb(R)^(I_1 times dots.h.c times I_N)$, it is natural to consider the function $X arrow.r.bar nabla^2 f (A) times_([N]) X$ for $X in bb(R)^(I_1 times dots.h.c times I_N)$. This gives us our definition of the operator norm on tensors.

#definition("Operator Norm")[
The operator norm of a half-symmetric tensor $A in bb(R)^((I_1 times dots.h.c times I_N)^2)$ (#ref(<eq-half-symmetric>, supplement: [Equation])) is defined as

#math.equation(block: true, numbering: "(1)", [ $ norm(A)_(upright("op")) = sup_(norm(X)_F = 1) norm(A times_([N]) X)_F . $ ])<eq-operator-norm>

In #ref(<eq-operator-norm>, supplement: [Equation]), $X in bb(R)^(I_1 times dots.h.c times I_N)$. Note that this definition agrees with the usual operator norm on matrices when $N = 1$.

] <def-operator-norm>
#theorem("Norm of an outer product")[
Let $T = A_1 times.circle dots.h.c times.circle A_N in bb(R)^((I_1 times dots.h.c times I_N)^2)$ where $A_n in bb(R)^(I_n times I_n)$ are symmetric matrices.

Then $T$ is half-symmetric and

$ norm(T)_(upright("op")) = product_(n = 1)^N norm(A_n)_(upright("op")) . $

] <thm-operator-norm-outer-product>
#block[
#callout(
body: 
[
According to how the outer product $times.circle$ is defined in #ref(<def-outer-product>, supplement: [Definition]), the product $A_1 times.circle dots.h.c times.circle A_N$ shown in #ref(<thm-operator-norm-outer-product>, supplement: [Theorem]) is really an element of $bb(R)^(I_1 times I_1 times dots.h.c times I_N times I_N)$. Note how the indexes are ordered differently than an element of $bb(R)^((I_1 times dots.h.c times I_N)^2) = bb(R)^(I_1 times dots.h.c times I_N times I_1 times dots.h.c times I_N)$. Correcting for this with explicit notation becomes cumbersome and would require tensor transposes, a new definition of an outer product, or reordering of indexes in the definition of a half-symmetric tensor. These can have knock-on effects to the definition of the Hessian, multi-mode product, and the operator norm.

To avoid the headache, the equality

$ T = A_1 times.circle dots.h.c times.circle A_N $

in #ref(<thm-operator-norm-outer-product>, supplement: [Theorem]) should be thought of as the following entry-wise equation

#math.equation(block: true, numbering: "(1)", [ $ T [i_1 \, dots.h \, i_N \, j_1 \, dots.h \, j_N] = A_1 [i_1 \, j_1] dots.h.c A_N [i_N \, j_N] . $ ])<eq-corrected-outer-product>

With the outer product understood as #ref(<eq-corrected-outer-product>, supplement: [Equation]), the results of #ref(<thm-operator-norm-outer-product>, supplement: [Theorem]) that $T$ is half-symmetric and its operator norm is the product of the operator norms of the constituent matrices is true.

]
, 
title: 
[
Warning
]
, 
background_color: 
rgb("#fcefdc")
, 
icon_color: 
rgb("#EB9113")
, 
icon: 
"⚠"
, 
body_background_color: 
white
)
]
#definition("Lipschitz Function")[
A function $g : bb(R)^(I_1 times dots.h.c times I_N) arrow.r bb(R)^(I_1 times dots.h.c times I_N)$ is $L$-Lipschitz when

$ norm(g (A) - g (B))_F lt.eq L norm(A - B)_F \, quad forall A \, B in bb(R)^(I_1 times dots.h.c times I_N) . $

We call the smallest such $L$ #emph[the] Lipschitz constant of $g$.

] <def-lipschitz>
#definition("Smooth Function")[
A differentiable function $f : bb(R)^(I_1 times dots.h.c times I_N) arrow.r bb(R)$ is $L$-smooth when its gradient $g = nabla f$ is $L$-Lipschitz.

] <def-smooth>
#theorem("Quadratic Smoothness")[
Let $f : bb(R)^(I_1 times dots.h.c times I_N) arrow.r bb(R)$ be a quadratic function

$ f (X) = 1 / 2 A (X \, X) + B (X) + C $

of $X$ with bilinear function $A : (bb(R)^(I_1 times dots.h.c times I_N))^2 arrow.r bb(R)$, linear function $B : bb(R)^(I_1 times dots.h.c times I_N) arrow.r bb(R)$, and constant $C in bb(R)^(I_1 times dots.h.c times I_N)$.

Then:

+ the Hessian $nabla^2 f$ is a constant function that evaluates to $nabla^2 f (X) = D$ at every point $X$ for some $D in bb(R)^((I_1 times dots.h.c times I_N)^2)$,
+ the tensor $D$ only depends on the bilinear function $A$ (and not on $B$ and $C$), and
+ the quadratic function $f$ is $L$-smooth with constant

$ L = norm(D)_(upright("op")) . $

] <thm-quadratic-smoothness>
== Common Decompositions
<sec-common-decompositions>
- Extensions of PCA/ICA/NMF to higher dimensions
- talk about the most popular Tucker, Tucker-n, CP
- other decompositions
  - high order SVD (see Kolda and Bader)
  - HOSVD (see Kolda, Shifted power method for computing tensor eigenpairs)

A tensor decomposition is a factorization of a tensor into multiple (usually smaller) tensors, that can be recombined into the original tensor. To make a common interface for decompositions, we make an abstract subtype of Julia's `AbstractArray`, and subtype `AbstractDecomposition` for our concrete tensor decompositions.

```julia
abstract type AbstractDecomposition{T, N} <: AbstractArray{T, N} end
```

Computationally, we can think of a generic decomposition as storing factors $(A \, B \, C \, . . .)$ and operations $(times_a \, times_b \, . . .)$ for combining them. This is what we do in BlockTensorFactorization.jl.

```julia
struct GenericDecomposition{T, N} <: AbstractDecomposition{T, N}
    factors::Tuple{Vararg{AbstractArray{T}}} # e.g. (A, B, C)
    contractions::Tuple{Vararg{Function}} # e.g. (×₁, ×₂)
end
# Y = A ×₁ B ×₂ C
array(G::GenericDecomposition) = multifoldl(contractions(G), factors(G))
```

The function `multifoldl` applies the given operations between each factor, from left to right.

```julia
function multifoldl(ops, args)
    @assert (length(ops) + 1) == length(args)
    x, xs... = args
    for (op, arg) in zip(ops, xs)
        x = op(x, arg)
    end
    return x
end
```

Different types of decompositions define different operations, and different "ranks" of the same decomposition specific the sizes of the factors used.

A commonly used family of decompositions can be derived from the Tucker decomposition.

#definition()[
A rank-$(R_1 \, dots.h \, R_N)$ Tucker decomposition of a tensor $Y in bb(R)^(I_1 times dots.h.c times I_N)$ produces $N$ matrices $A_n in bb(R)^(I_n times R_n)$, $n in [N]$, and core tensor $B in bb(R)^(R_1 times dots.h.c times R_N)$ such that

#math.equation(block: true, numbering: "(1)", [ $ Y [i_1 \, dots.h \, i_N] = sum_(r_1 = 1)^(R_1) dots.h sum_(r_N = 1)^(R_N) A_1 [i_1 \, r_1] dots.h.c A_r [i_N \, r_N] B [r_1 \, dots.h \, r_N] $ ])<eq-tucker>

entry-wise. More compactly, this decomposition can be written using the $n$-mode product, or with double brackets

#math.equation(block: true, numbering: "(1)", [ $ Y = B times_1 A_1 times_2 dots.h times_N A_N = B times.big_n A_n = lr(bracket.l.double B \; A_1 \, dots.h \, A_N bracket.r.double) . $ ])<eq-tucker-product>

] <def-tucker-decomposition>
The #emph[Tucker Product] defined by #ref(<eq-tucker-product>, supplement: [Equation]) is implemented in BlockTensorFactorization.jl with `tuckerproduct(B, (A1, ..., AN))` and computes $ B times.big_n A_n = lr(bracket.l.double B \; A_1 \, dots.h \, A_N bracket.r.double) . $

It can also optionally "exclude" one of the matrix factors with the call `tuckerproduct(B, (A1, ..., AN); exclude=n)` to compute

$ B times.big_(m eq.not n) A_m = lr(bracket.l.double B \; A_1 \, dots.h \, A_(n - 1) \, upright("id")_(R_n) \, A_(n + 1) \, dots.h \, A_N bracket.r.double) . $

```julia
function tuckerproduct(core, matrices; exclude=nothing)
    N = ndims(core)
    if isnothing(exclude)
        return multifoldl(tucker_contractions(N), (core, matrices...))
    else
        return multifoldl(getnotindex(tucker_contractions(N), exclude), (core, getnotindex(matrices, exclude)...))
    end
end

tucker_contractions(N) = Tuple((B, A) -> nmode_product(B, A, n) for n in 1:N)
```

Sometimes we write $A_0 = B$ to ease notation, and suggest the "zeroth" factor of the tucker decomposition is the core tensor $B$. In the special case when $N = 3$, we can visualize Tucker decomposition as multiplying the core tensor by matrices on all three sides as shown in #ref(<fig-tucker>, supplement: [Figure]).

#figure([
#box(image("figure/tucker_decomposition_order_3.png"))
], caption: figure.caption(
position: bottom, 
[
Tucker factorization of a $3$rd order tensor $Y$.
]), 
kind: "quarto-float-fig", 
supplement: "Figure", 
)
<fig-tucker>


Setting all the matrices of a Tucker decomposition to the identity matrix but the first gives the Tucker-$1$ decomposition.

#definition()[
A rank-$R$ Tucker-$1$ decomposition of a tensor $Y in bb(R)^(I_1 times dots.h.c times I_N)$ produces a matrix $A in bb(R)^(I_1 times R)$, and core tensor $B in bb(R)^(R times I_2 times dots.h.c times I_N)$ such that

#math.equation(block: true, numbering: "(1)", [ $ Y [i_1 \, dots.h \, i_N] = sum_(r = 1)^R A [i_1 \, r] B [r \, i_2 \, dots.h \, i_N] $ ])<eq-tucker-1>

entry-wise or more compactly,

$ Y = A B = B times_1 A = lr(bracket.l.double B \; A bracket.r.double) . $

] <def-tucker-1-decomposition>
Note we extend the usual definition of matrix-matrix multiplication

$ (A B) [i \, j] = sum_(r = 1)^R A [i \, r] B [r \, j] $

to tensors $B$ in the compact notation for Tucker-1 decomposition $Y = A B$.

More generally, any number of matrices can be set to the identity matrix giving the Tucker-$n$ decomposition.

#definition()[
A rank-$(R_1 \, dots.h \, R_n)$ Tucker-$n$ decomposition of a tensor $Y in bb(R)^(I_1 times dots.h.c times I_N)$ produces $n$ matrices $A_1 \, dots.h \, A_n$, and core tensor $B in bb(R)^(R_1 times dots.h.c times R_n times I_(n + 1) times dots.h.c times I_N)$ such that

#math.equation(block: true, numbering: "(1)", [ $ Y [i_1 \, dots.h \, i_N] = sum_(r_1 = 1)^(R_1) dots.h sum_(r_N = 1)^(R_n) A_1 [i_1 \, r_1] dots.h.c A_n [i_N \, r_n] B [r_1 \, dots.h \, r_n \, i_(n + 1) \, dots.h \, i_N] $ ])<eq-tucker-n>

entry-wise, or compactly written in the following three ways, $ Y & = B times_1 A_1 times_2 dots.h times_n A_n times_(n + 1) upright(i d)_(I_(n + 1)) times_(n + 2) dots.h times_N upright(i d)_(I_N)\
Y & = B times_1 A_1 times_2 dots.h times_n A_n\
Y & = lr(bracket.l.double B \; A_1 \, dots.h \, A_n bracket.r.double) . $

] <def-tucker-n-decomposition>
Lastly, if we set the core tensor $B$ to the identity tensor $upright(i d)_R$, we obtain the #strong[can];onical #strong[decomp];osition/#strong[para];llel #strong[fac];tors model (CANDECOMP/PARAFAC or CP for short).

#definition()[
A rank-$R$ CP decomposition of a tensor $Y in bb(R)^(I_1 times dots.h.c times I_N)$ produces $N$ matrices $A_n in bb(R)^(I_n times R)$, such that

#math.equation(block: true, numbering: "(1)", [ $ Y [i_1 \, dots.h \, i_N] = sum_(r = 1)^R A_1 [i_1 \, r] dots.h.c A_r [i_N \, r] $ ])<eq-cp>

entry-wise. More compactly, this decomposition can be written using the $n$-mode product, or with double brackets

$ Y = upright(i d)_R times_1 A_1 times_2 dots.h times_N A_N = upright(i d)_R times.big_n A_n = lr(bracket.l.double A_1 \, dots.h \, A_N bracket.r.double) . $

] <def-cp-decomposition>
Note CP decomposition is sometimes referred to as Kruskal decomposition, and requires the core only be diagonal (and not necessarily identity) and the factors $A_n$ have normalized columns $norm(A_n [: \, r])_2 = 1$.

Other factorization models are used that combine aspects of CP and Tucker decomposition @kolda_TensorDecompositionsApplications_2009, are specialized for order $3$ tensors @qi_TripleDecompositionTensor_2020@wu_manifold_2022, or provide alternate decomposition models entirely like tensor-trains @oseledets_tensor-train_2011. But the (full) Tucker, and its special cases Tucker-$n$, and CP decomposition are most commonly used extensions of the low-rank matrix factorization to tensors. These factorizations are summarized in #ref(<tbl-tensor-factorizations>, supplement: [Table]).

#figure([
#table(
  columns: (15%, 25%, 35%, 25%),
  align: (auto,auto,auto,auto,),
  table.header([Name], [Bracket Notation], [$n$-mode Product], [Entry-wise],),
  table.hline(),
  [Tucker], [$lr(bracket.l.double A_0 \; A_1 \, dots.h \, A_N bracket.r.double)$], [$A_0 times_1 A_1 times_2 dots.h times_N A_N$], [#ref(<eq-tucker>, supplement: [Equation])],
  [Tucker-$1$], [$lr(bracket.l.double A_0 \; A_1 bracket.r.double)$], [$A_0 times_1 A_1$], [#ref(<eq-tucker-1>, supplement: [Equation])],
  [Tucker-$n$], [$lr(bracket.l.double A_0 \; A_1 \, dots.h \, A_n bracket.r.double)$], [$A_0 times_1 A_1 times_2 dots.h times_n A_n$], [#ref(<eq-tucker-n>, supplement: [Equation])],
  [CP], [$lr(bracket.l.double A_1 \, dots.h \, A_N bracket.r.double)$], [$upright(i d)_R times_1 A_1 times_2 dots.h times_N A_N$], [#ref(<eq-cp>, supplement: [Equation])],
)
], caption: figure.caption(
position: top, 
[
Summary of common tensor factorizations. Here, $N$ is the order of the factorized tensor.
]), 
kind: "quarto-float-tbl", 
supplement: "Table", 
)
<tbl-tensor-factorizations>


TODO add discussion on other decompositions - high order SVD (see Kolda and Bader) - HOSVD (see Kolda, Shifted power method for computing tensor eigenpairs)

Tensor decompositions are not nessisarily unique. It should be clear that scaling one factor by $x eq.not 0$ and dividing another by $x$ yields the same original tensor. Furthermore, fibres and slices can be permuted without affecting the the original tensor. Up to these manipulations, for a fixed rank, there exist criteria that ensures their decompositions are unique @kolda_TensorDecompositionsApplications_2009@kruskal_three-way_1977@bhaskara_uniqueness_2014.

=== Representing Tucker Decompositions
<representing-tucker-decompositions>
There are implemented in BlockTensorFactorization.jl and can be called, for a third order tensor, with `Tucker((B, A₁, A₂, A₃))`, `Tucker1((B, A₁))`, and `CPDecomposition((A₁, A₂, A₃))`. These Julia `structs` store the tensor in its factored form. We could define the contractions for these types and use the common interface provided by `array`, but it turns out we can reconstruct the whole tensor more efficiently. If the recombined tensor or particular entries are requested, Julia dispatches on the type of decomposition and calls a particular method of `array` or `getindex`. The implementations for efficient array construction and index access are provided below.

```julia
array(T::Tucker) = multifoldl(tucker_contractions(ndims(T)), factors(T))
tucker_contractions(N) = Tuple((G, A) -> nmode_product(G, A, n) for n in 1:N)
```

TODO add getindex method for Tucker type

```julia
function array(T::Tucker1)
    B, A = factors(T)
    return B ×₁ A
end

function getindex(T::Tucker1, I::Vararg{Int})
    B, A = factors(T)
    i, J... = I # (i, J) = (I[1], I[begin+1:end])
    return (@view A[i, :]) ⋅ view(B, :, J...)
end
```

```julia
array(CPD::CPDecomposition) =
  mapreduce(vector_outer, +, zip((eachcol.(factors(CPD)))...))

vector_outer(v) = reshape(kron(reverse(v)...),length.(v))

getindex(CPD::CPDecomposition, I::Vararg{Int}) =
  sum(reduce(.*, (@view f[i,:]) for (f,i) in zip(factors(CPD), I)))
```

== Tensor rank
<sec-tensor-rank>
- tensor rank
- constrained rank (nonnegative etc.)

The rank of a matrix $Y in bb(R)^(I times J)$ can be defined as the smallest $R in bb(Z)_(+)$ such that there exists an exact factorization $Y = A B$ for some $A in bb(R)^(I times R)$ and $B in bb(R)^(R times J)$.

Although this can be extended to higher order tensors, we must specify under which factorization model we are using. For example, the #emph[CP-rank] $R$ of a tensor $Y$ is the smallest such $R$ that omits an exact CP decomposition of $Y$.

#definition()[
The CP rank of a tensor $Y in bb(R)^(I_1 times dots.h.c times I_N)$ is the smallest $R$ such that there exist factors $A_n in bb(R)^(I_n times R)$ and $Y = lr(bracket.l.double A_1 \, dots.h \, A_N bracket.r.double)$, $ upright("rank")_(upright("CP")) (Y) = min {R mid(bar.v) exists A_n in bb(R)^(I_n times R) \, thin n in [N] quad upright("s.t.") quad Y = lr(bracket.l.double A_1 \, dots.h \, A_N bracket.r.double)} . $

] <def-cp-rank>
In a similar way, we can define the #emph[Tucker-1-rank] $R$.

#definition()[
The Tucker-1 rank of a tensor $Y in bb(R)^(I_1 times dots.h.c times I_N)$ is the smallest $R$ such that there exist factors $A in bb(R)^(I_1 times R)$ and $B in bb(R)^(R times I_2 times dots.h.c times I_N)$ where $Y = A B$

$ upright("rank")_(upright("Tucker-1")) (Y) = min {R mid(bar.v) exists A_n in bb(R)^(I_n times R) \, B in bb(R)^(R times I_2 times dots.h.c times I_N) thin quad upright("s.t.") quad Y = A B} $

] <def-tucker-1-rank>
By convention, we say the zero tensor $Y = 0$ has rank $0$.

For the Tucker and Tucker-$n$ decompositions, we instead call a particular factorization #strong[a] rank-$(R_1 \, dots.h \, R_N)$ Tucker factorization or #strong[a] rank-$(R_1 \, dots.h \, R_n)$ Tucker-$n$ factorization, rather than #strong[the] CP- or Tucker-$1$-rank of a tensor or #strong[the] rank of a matrix.

One reason CP and Tucker-$1$ only need a single rank $R$ can be explained by considering the case when the order of the tensor $N = 2$ (matrices). The two factorizations become equivalent and are equal to low-rank $R$ matrix factorization $Y = A B$. In fact, Tucker-$1$ is always equivalent to a low-rank matrix factorization, if you consider a flattening of the tensor to arrange the entries as a matrix.

The idea of tensor rank can be generalized further to constrained rank. These are the smallest rank $R$ such that the factors in the decomposition obey the given set of constraints.

For example, the nonnegative Tucker-1 rank is defined as $ upright("rank")_(upright("Tucker-1"))^(+) (Y) = min {R mid(bar.v) exists A_n in bb(R)_(+)^(I_n times R) \, B in bb(R)_(+)^(R times I_2 times dots.h.c times I_N) thin quad upright("s.t.") quad Y = A B} . $

More restrictive constraints increase the rank of the tensor since there is less freedom in selecting the factors.

Most tensor decomposition algorithms require the rank as input \[CITE\] since calculating the rank of the tensor can be NP-hard in general @vavasis_complexity_2010. For applications where the rank is not known a priori, a common strategy is to attempt a decomposition for a variety of ranks, and select the model with smallest rank that still achieves good fit between the factorization and the original tensor. See section #ref(<sec-estimating-tensor-rank>, supplement: [Section]) for an implementation of this strategy.

= Computing Decompositions
<computing-decompositions>
- Given a data tensor and a model, how do we fit the model?

Many tensor decompositions algorithms exist in the literature. Usually, they cyclically (or in a random order) update factors until their reconstruction satisfies some convergence criterion. The base algorithm described in #ref(<sec-base-algorithm>, supplement: [Section]) provides flexible framework for wide class of constrained tensor factorization problems. This framework was selected based on empirical observations where it outperforms other similar algorithms, and has also been observed in the literature @xu_BlockCoordinateDescent_2013.

== Optimization Problem
<optimization-problem>
- Least squares (can use KL, 1 norm, etc.)

Ideally, we would be given a data tensor $Y$ and decomposition model, and compute an exact factorization of $Y$ into its factors. Because there is often measurement, numerical, or modeling error, an exact factorization of $Y$ for a particular rank may not exist. To over come this, we instead try to fit the model to the data. Let $X$ be the reconstruction of factors $A_1 \, dots.h \, A_N$ according to some decomposition for a fixed rank. We assume we know the size of the factors $A_1 \, dots.h \, A_N$ and how they are combined to produce a tensor the same size of $Y$, i.e.~the map $g : (A_1 \, dots.h \, A_N) arrow.r.bar X$.

There are many loss functions that can be used to determine how close the model $X$ is to the data $Y$. In principle, any distance or divergence $d (Y \, X)$ could be used. We use the $L_2$ loss or least-squares distance between the tensors $norm(X - Y)_F^2$, but other losses are used for tensor decomposition in practice such as the KL divergence \[CITE\].

The main optimization we must solve is now given.

#definition()[
The constrained least-squares tensor factorization problem is to solve

#math.equation(block: true, numbering: "(1)", [ $ min_(A_1 \, dots.h \, A_N) 1 / 2 norm(g (A_1 \, dots.h \, A_N) - Y)_F^2 quad upright("s.t.") quad (A_1 \, dots.h \, A_N) in cal(C)_1 times dots.h.c times cal(C)_N $ ])<eq-constrained-least-squares>

for a given data tensor $Y$, constraints $cal(C)_1 \, dots.h \, cal(C)_N$, and decomposition model $g$ with fixed rank.

] <def-constrained-least-squares>
Note the problem would have the same solutions as simply using the objective $norm(g (A_1 \, dots.h \, A_N) - Y)$ without squaring and dividing by $2$. We define the objective in #ref(<eq-constrained-least-squares>, supplement: [Equation]) to make computing the function value and gradients faster.

== Base algorithm
<sec-base-algorithm>
- Use Block Coordinate Descent / Alternating Proximal Descent
  - do #emph[not] use alternating least squares (slower for unconstrained problems, no closed form update for general constrained problems)

Let $f (A_1 \, dots.h \, A_N) := 1 / 2 norm(g (A_1 \, dots.h \, A_N) - Y)_F^2$ be the objective function we wish to minimize in #ref(<eq-constrained-least-squares>, supplement: [Equation]). Following Xu and Yin @xu_BlockCoordinateDescent_2013, the general approach we take to minimize $f$ is to apply block coordinate descent using each factor as a different block. Let $A_n^t$ be the $t$th iteration of the $n$th factor, and let

$ f_n^t (A_n) := 1 / 2 norm(g (A_1^(t + 1) \, dots.h \, A_(n - 1)^(t + 1) \, A_n \, A_(n + 1)^t \, dots.h \, A_N^t) - Y)_F^2 $

be the (partially updated) objective function at iteration $t$ for factor $n$.

Given initial factors $A_1^0 \, dots.h \, A_N^0$, we cycle through the factors $n in [N]$ and perform the update

$ A_n^(t + 1) arrow.l arg min_(A_n in cal(C)_n) ⟨nabla f_n^t (A_n^t) \, A_n - A_n^t⟩ + L_n^t / 2 norm(A_n - A_n^t)_F^2 \, $

for $t = 1 \, 2 \, dots.h$ until some convergence criterion is satisfied (see #ref(<sec-convergence-criteria>, supplement: [Section])).

This implicit update has the #emph[projected gradient descent] closed form solution for convex constraints $cal(C)_n$,

#math.equation(block: true, numbering: "(1)", [ $ A_n^(t + 1) arrow.l P_(cal(C)_n) (A_n^t - 1 / L_n^t nabla f_n^t (A_n^t)) . $ ])<eq-proximal-explicit>

We typically choose $L_n^t$ to be the Lipschitz constant of $nabla f_n^t$, since it is a sufficient condition to guarantee $f_n^t (A_n^(t + 1)) lt.eq f_n^t (A_n^t)$, but other stepsizes can be used in theory @nesterov_NonlinearOptimization_2018[Sec. 1.2.3].

?ASIDE? To write $nabla f_n^t$, we have assumed (block) differentiability of the decomposition model $g$. In practice, most decompositions are "block-linear" (freeze all factors but one and you have a linear function) and in rare cases are "block-affine". "block-affine" is enough to ensure $f_n^t$ is convex (i.e.~$f$ is "block-convex") so the updates #ref(<eq-proximal-explicit>, supplement: [Equation]) converge to a Nash equilibrium (block minimizer).

=== High level code
<high-level-code>
To ensure the code stays flexible, the main algorithm of BlockTensorFactorization.jl, `factorize`, is defined at a very high level.

```julia
factorize(Y; kwargs...) =
    _factorize(Y; (default_kwargs(Y; kwargs...))...)

"""
Inner level function once keyword arguments are set
"""
function _factorize(Y; kwargs...)
    decomposition, previous, updateprevious!, parameters, updateparameters!,
    update!, stats_data, getstats, converged, kwargs = initialize(Y, kwargs)

    while !converged(stats_data; kwargs...)
        # Usually one cycle of updates through each factor in the decomposition
        update!(decomposition; parameters...)

        # This could be the next stepsize or other info used by update!
        updateparameters!(parameters, decomposition, previous)

        push!(stats_data,
            getstats(decomposition, Y, previous, parameters, stats_data))

        # Update one or two previous iterates. For example, used for momentum
        updateprevious!(previous, parameters, decomposition)
    end

    kwargs = postprocess!(decomposition, Y, previous, parameters, stats_data, updateparameters!, getstats, kwargs)

    return decomposition, stats_data, kwargs
end
```

The magic of the code is in defining the functions at runtime for a particular decomposition requested, from a reasonable set of default keyword arguments. This is discussed further in #ref(<sec-flexibility>, supplement: [Section]).

=== Computing Gradients
<sec-gradient-computation>
- Use Auto diff generally
- But hand-crafted gradients and Lipschitz calculations #emph[can] be faster (e.g.~symmetrized slice-wise dot product)

Generally, we can use automatic differentiation on $f$ to compute gradients. Some care needs to be taken otherwise the forward or backwards pass will have to be recompiled every iteration since the factors are updated every iteration.

But for Tucker decompositions, we can compute gradients faster than what an automatic differentiation scheme would give, by taking advantage of symmetry and other computational shortcuts.

Starting with the Tucker-1 decomposition (#ref(<def-tucker-1-decomposition>, supplement: [Definition])), we would like to compute $nabla_B f (B \, A)$ and $nabla_A f (B \, A)$ for $f (B \, A) = 1 / 2 norm(A B - Y)_F^2$ for a given input $Y$. We have the gradient

#math.equation(block: true, numbering: "(1)", [ $ nabla_B f (B \, A) = A^tack.b (A B - Y) = (B times_1 A - Y) times_1 A^tack.b $ ])<eq-tucker-1-gradient-1>

by chain rule, but it is more efficient to calculate the gradient as

#math.equation(block: true, numbering: "(1)", [ $ nabla_B f (B \, A) = (A^tack.b A) B - A^tack.b Y = B times_1 (A^tack.b A) - Y times_1 A^tack.b . $ ])<eq-tucker-1-gradient-2>

#footnote[Seeing #ref(<eq-tucker-1-gradient-1>, supplement: [Equation]) and #ref(<eq-tucker-1-gradient-2>, supplement: [Equation]) written using the $1$-mode product shows how it is "backwards" to normal matrix-matrix multiplication.];For $A in bb(R)^(I times R)$, $B in bb(R)^(R times J times K)$, and $Y in bb(R)^(I times J times K)$, #ref(<eq-tucker-1-gradient-1>, supplement: [Equation]) requires

$ underbrace(2 I J K R, A B - Y) + underbrace(I J K (2 I - 1), A^tack.b (A B - Y)) tilde.op 2 I J K R + 2 I^2 J K $ floating point operations (FLOPS) whereas #ref(<eq-tucker-1-gradient-2>, supplement: [Equation]) only uses

$ underbrace(frac(R (R + 1), 2) (2 I - 1), A^tack.b A) + underbrace(R J K (2 I - 1), A^tack.b Y) + underbrace(2 R^2 J K, (A^tack.b A) B - (A^tack.b Y)) tilde.op 2 I J K R + 2 R^2 J K + I R^2 $

FLOPS#footnote[Note we have the smaller factor $R (R + 1) \/ 2$ and not the expected $R^2$ number of entries needed to compute $A^tack.b A$. The product is a symmetric matrix so only the upper or lower triangle of entries needs to be computed.]. So for small ranks $R lt.double I$, #ref(<eq-tucker-1-gradient-2>, supplement: [Equation]) is cheaper.

A similar story can be said about $nabla_A f (B \, A)$ which is most efficiently computed as

$ nabla_A f (B \, A) = A (B dot.op_1 B) - Y dot.op_1 B . $

#block[
#callout(
body: 
[
For the family of Tucker decompositions, the objective function $f$ is "block-quadratic" with respect to the factors. This means the gradient with respect to a factor is an affine function of that factor. This is exactly what we see in #ref(<eq-tucker-1-gradient-2>, supplement: [Equation]) where $B$ is multiplied by the "slope" $A^tack.b A$ plus a shift of $- Y times_1 A^tack.b$.

]
, 
title: 
[
Note
]
, 
background_color: 
rgb("#dae6fb")
, 
icon_color: 
rgb("#0758E5")
, 
icon: 
"❕"
, 
body_background_color: 
white
)
]
The associated implementation with BlockTensorFactorization.jl is shown below. We define a `make_gradient` which takes the decomposition, factor index `n`, and data tensor `Y`, and creates a function that computes the gradient for the same type of decomposition. This lets us manipulate the function that computes the gradient, rather than just the computed gradient.

```julia
function make_gradient(T::Tucker1, n::Integer, Y::AbstractArray; objective::L2, kwargs...)
    if n==0 # the core is the zeroth factor
        function gradient0(X::Tucker1; kwargs...)
            (B, A) = factors(X)
            AA = A'A
            YA = Y×₁A'
            grad = B×₁AA - YA
            return grad
        end
        return gradient0
    elseif n==1 # the matrix is the first factor
        function gradient1(X::Tucker1; kwargs...)
            (B, A) = factors(X)
            BB = slicewise_dot(B, B)
            YB = slicewise_dot(Y, B)
            grad = A*BB - YB
            return grad
        end
        return gradient1
    else
        error("No $(n)th factor in Tucker1")
    end
end
```

Similarly, we also have special methods for the Tucker and CP Decomposition.

The gradient with respect to the core for a full Tucker factorization is

$ nabla_B f (B \, A_1 \, dots.h \, A_N) = B times.big_n A_n^tack.b A_n - Y times.big_n A_n^tack.b \, $

and the gradient with respect to the matrix factor $A_n$ is

$ nabla_(A_n) f (B \, A_1 \, dots.h \, A_N) = A_n (tilde(X)_n dot.op_n tilde(X)_n) - Y dot.op_n tilde(X)_n $

where

$ tilde(X)_n = (B times.big_(m eq.not n) A_m) = lr(bracket.l.double B \; A_1 \, dots.h \, A_(n - 1) \, upright("id")_(R_n) \, A_(n + 1) \, dots.h \, A_N bracket.r.double) . $

```julia
function make_gradient(T::Tucker, n::Integer, Y::AbstractArray; objective::L2, kwargs...)
    N = ndims(T)
    if n==0 # the core is the zeroth factor
        function gradient_core(X::AbstractTucker; kwargs...)
            B = core(X)
            matrices = matrix_factors(X)
            gram_matrices = map(A -> A'A, matrices) # gram matrices AA = A'A,
                                                    # BB = B'B, ...
            grad = tuckerproduct(B, gram_matrices)
                 - tuckerproduct(Y, adjoint.(matrices))
            return grad
        end
        return gradient_core

    elseif n in 1:N # the matrix factors start at m=1
        function gradient_matrix(X::AbstractTucker; kwargs...)
            B = core(X)
            matrices = matrix_factors(X)
            Aₙ = factor(X, n)
            X̃ₙ = tuckerproduct(B, matrices; exclude=n)
            grad = Aₙ * slicewise_dot(X̃ₙ, X̃ₙ; dims=n)
                   - slicewise_dot(Y, X̃ₙ; dims=n)
            return grad
        end
        return gradient_matrix

    else
        error("No $(n)th factor in Tucker")
    end
end
```

For the CP Decomposition, we can simply treat the core as $B = upright("id")_R$ and compute the gradient with respect to the matrix factors similarly to the Tucker decomposition:

$ nabla_(A_n) f (A_1 \, dots.h \, A_N) = A_n (tilde(X)_n dot.op_n tilde(X)_n) - Y dot.op_n tilde(X)_n $

where

$ tilde(X)_n = (upright("id")_R times.big_(m eq.not n) A_m) = lr(bracket.l.double upright("id")_R \; A_1 \, dots.h \, A_(n - 1) \, upright("id")_R \, A_(n + 1) \, dots.h \, A_N bracket.r.double) . $

```julia
function make_gradient(T::CPDecomposition, n::Integer, Y::AbstractArray; objective::L2, kwargs...)
    N = ndims(T)
    if n in 1:N # the matrix factors start at m=1
        function gradient_matrix(X::AbstractTucker; kwargs...)
            B = core(X)
            matrices = matrix_factors(X)
            Aₙ = factor(X, n)
            X̃ₙ = tuckerproduct(B, matrices; exclude=n)
            grad = Aₙ * slicewise_dot(X̃ₙ, X̃ₙ; dims=n)
                   - slicewise_dot(Y, X̃ₙ; dims=n)
            return grad
        end
        return gradient_matrix

    else
        error("No $(n)th factor in Tucker")
    end
end
```

=== Computing Lipschitz Step-sizes
<sec-lipschitz-computation>
Similar to automatic differentiation, there exist "automatic Lipschitz" calculations to upper bound the Lipschitz constant of a function @virmaux_lipschitz_2018.

For the family of Tucker decompositions, we can compute the Lipschitz constants of the gradient efficiently similar to how we compute the gradient in #ref(<sec-gradient-computation>, supplement: [Section]) with the following corollaries of #ref(<thm-quadratic-smoothness>, supplement: [Theorem]).

#corollary()[
Let $B in bb(R)^(R_1 times dots.h.c times R_N)$, $A_m in bb(R)^(I_m times R_m)$, and $Y in bb(R)^(I_1 times dots.h.c times I_N)$. The function

$ f (A) = 1 / 2 norm([B \; A_1 \, dots.h \, A_(n - 1) \, A \, A_(n + 1) \, dots.h \, A_N] - Y)_F^2 $

is quadratic, and $L$-smooth with constant

$ L_(A_n) = norm(tilde(X)_n dot.op_n tilde(X)_n)_(upright("op")) $

where

$ tilde(X)_n = (B times.big_(m eq.not n) A_m) = lr(bracket.l.double B \; A_1 \, dots.h \, A_(n - 1) \, upright("id")_(R_n) \, A_(n + 1) \, dots.h \, A_N bracket.r.double) . $

#block[
#emph[Proof]. The result follows from #ref(<thm-operator-norm-outer-product>, supplement: [Theorem]) and #ref(<thm-quadratic-smoothness>, supplement: [Theorem]).

]
] <cor-least-squares-matrix>
#corollary()[
Let $A_n in bb(R)^(I_n times R_n)$, and $Y in bb(R)^(I_1 times dots.h.c times I_N)$. The function

$ f (B) = 1 / 2 norm([B \; A_1 \, dots.h \, A_N] - Y)_F^2 $

is quadratic, and $L$-smooth with constant

$ L_B = product_(n = 1)^N norm(A_n^tack.b A_n)_(upright("op")) . $

#block[
#emph[Proof]. The result follows from #ref(<thm-operator-norm-outer-product>, supplement: [Theorem]) and #ref(<thm-quadratic-smoothness>, supplement: [Theorem]).

]
] <cor-least-squares-core>
This yields the following efficient implementations.

#block[
#callout(
body: 
[
It is tempting to use the identity $norm(A^tack.b A)_(upright("op")) = norm(A)_(upright("op"))^2$ to calculate the Lipschitz constant without forming $A^tack.b A$. For tall dense matrices, using this identity is slower and more memory intensive as of Julia 1.11.2. See #link("https://github.com/JuliaLang/LinearAlgebra.jl/issues/1185")[LinearAlgebra.jl issue 1185] on Github.

]
, 
title: 
[
Note
]
, 
background_color: 
rgb("#dae6fb")
, 
icon_color: 
rgb("#0758E5")
, 
icon: 
"❕"
, 
body_background_color: 
white
)
]
```julia
function make_lipschitz(T::Tucker, n::Integer, Y::AbstractArray; objective::L2, kwargs...)
    N = ndims(T)
    if n==0 # the core is the zeroth factor
        function lipschitz_core(X::AbstractTucker; kwargs...)
            return prod(A -> opnorm(A'A), matrix_factors(X))
        end
        return lipschitz_core

    elseif n in 1:N # the matrix is the zeroth factor
        function lipschitz_matrix(X::AbstractTucker; kwargs...)
            B = core(X)
            matrices = matrix_factors(X)
            X̃ₙ = tuckerproduct(B, matrices; exclude=n)
            return opnorm(slicewise_dot(X̃ₙ, X̃ₙ; dims=n))
        end
        return lipschitz_matrix

    else
        error("No $(n)th factor in Tucker")
    end
end
```

In the case of Tucker decomposition, the Lipschitz constants simplify to

$ L_B = norm(A^tack.b A)_(upright("op")) \, quad L_B = norm(B dot.op_1 B)_(upright("op")) . $

```julia
function make_lipschitz(T::Tucker1, n::Integer, Y::AbstractArray; objective::L2, kwargs...)
    if n==0 # the core is the zeroth factor
        function lipschitz0(X::Tucker1; kwargs...)
            A = matrix_factor(X, 1)
            return opnorm(A'A)
        end
        return lipschitz0

    elseif n==1 # the matrix is the zeroth factor
        function lipschitz1(X::Tucker1; kwargs...)
            B = core(X)
            return opnorm(slicewise_dot(B, B))
        end
        return lipschitz1

    else
        error("No $(n)th factor in Tucker1")
    end
end
```

Lastly, for CP decomposition, the Lipschitz constants for the matrices can be calculated similarly to the Tucker decomposition.

```julia
function make_lipschitz(T::CPDecomposition, n::Integer, Y::AbstractArray; objective::L2, kwargs...)
    N = ndims(T)
    if n in 1:N
        function lipschitz_matrix(X::CPDecomposition; kwargs...)
            id = core(X)
            matrices = matrix_factors(X)
            X̃ₙ = tuckerproduct(id, matrices; exclude=n)
            return opnorm(slicewise_dot(X̃ₙ, X̃ₙ; dims=n))
        end
        return lipschitz_matrix

    else
        error("No $(n)th factor in CPDecomposition")
    end
end
```

=== Estimating Tensor Rank
<sec-estimating-tensor-rank>
In many applications, the rank of the input tensor $Y$ may not be known. In the case of CP and Tucker-1 decompositions, there are known bounds on the rank.

#lemma()[
Given a tensor $Y in bb(R)^(I_1 times dots.h.c times I_N)$, we have the following bounds on the Tucker-1 and CP rank @kolda_TensorDecompositionsApplications_2009[Sec. 3.1].

$ 0 lt.eq upright("rank")_(upright("CP")) (Y) lt.eq min_n product_(m eq.not n) I_m = min (I_2 dots.h.c I_N \, I_1 I_3 dots.h.c I_N \, dots.h \, I_1 dots.h.c I_(N - 1)) \, $

and $ 0 lt.eq upright("rank")_(upright("Tucker-1")) (Y) lt.eq min (I_1 \, product_(n = 2)^N I_n) = min (I_1 \, I_2 dots.h.c I_N) . $

We have a rank of $0$ if and only if $Y = 0$ is the zero tensor.

] <lem-tensor-rank-bounds>
Both of these bounds are natural extensions of the typical matrix rank bounds. For a matrix $Y in bb(R)^(I times J)$, we know

$ 0 lt.eq upright("rank") (Y) lt.eq min (I \, J) . $

In the case that we try to factorize a matrix $Y$ at a rank $R gt.eq upright("rank") (Y)$, we should still be able to achieve a final objective of zero $0 = 1 / 2 norm(A^(\*) B^(\*) - Y)_F^2$, and at any rank $R < upright("rank") (Y)$, the objective is positive and decreasing. This fact extends to CP and Tucker-1 decompositions. So a naive strategy for finding the rank of a tensor $Y$, would be first check if it is rank zero, and then try factorizing $Y$ at incrementally larger ranks these two types of decompositions until the final objective hits zero.

This approach has two problems. With real work data, the input tensor is often a noisy version of a low rank tensor $Y = Y_(upright("clean")) + Z$. Noise is often full rank and would cause the input tensor $Y$ to be the maximum rank @vershynin_HighDimensionalProbability_2018. Secondly, even if we had a perfectly low rank tensor $Y$, our algorithm at best converges to the optimal solution in the limit. So any finite stopping point has a positive objective.

A simple fix would be to extend the concept of numerical rank (also called $epsilon.alt$-rank) to tensors.#footnote[See #link("https://mpf-optimization-laboratory.github.io/opt-blog/posts/epsilon-rank/") for an in-depth discussion on this topic. TODO move some proofs to this document.] This serves as an approximation for the rank where taking $epsilon.alt arrow.r 0^(+)$ would return the usual definition of the rank.

#definition()[
The $epsilon.alt$-rank @golubRosetakDocumentRank1977 of a matrix $A$ is the smallest rank obtainable by an $epsilon.alt$-perturbation of the matrix $A + E$: #math.equation(block: true, numbering: "(1)", [ $ upright(r a n k)_epsilon.alt (A) = min_(norm(E) lt.eq epsilon.alt) upright(r a n k) (A + E) . $ ])<eq-epsilon-rank>

] <def-epsilon-rank>
Here we use $norm(E) = max_(lr(bar.v.double v bar.v.double)_2 = 1) norm(E v)_2$ to denote the the operator norm.

This definition is #emph[stable] in the sense that adding a small amount of noise to a matrix does not effect the $epsilon.alt$-rank. This is unlike the traditional rank where adding noise to a low rank matrix can turn it into a full rank matrix, even for an "$epsilon.alt$" amount of noise. This is clear from #ref(<prp-singular-values>, supplement: [Proposition]) which also offers a practical method for computing $upright(r a n k)_epsilon.alt (A)$.

#proposition("Singular Value Characterization")[
The $epsilon.alt$-rank of $A$ is equal to the number of singular values of $A$ strictly bigger than $epsilon.alt$.

] <prp-singular-values>
The following new result, #ref(<thm-rank-recovery>, supplement: [Theorem]), shows that we can use the $epsilon.alt$-rank to compute the true rank of a matrix under some additive noise $B$ with the right choice of $epsilon.alt$.

#theorem("Rank recovery")[
Let $A \, B in bb(R)^(m times n)$ where $norm(B)_2 < sigma_r \/ 2$. Then, $ upright(r a n k)_epsilon.alt (A + B) = upright(r a n k) (A) = r $

for all $epsilon.alt$ between $norm(B)_2 lt.eq epsilon.alt < sigma_r \/ 2 .$ Here, we list the singular values of $A$ in non-increasing order: $sigma_1 gt.eq dots.h gt.eq sigma_r > sigma_(r + 1) = dots.h = sigma_(min (m \, n)) = 0$.

] <thm-rank-recovery>
Note there are no assumptions on the noise $B$. It could be fixed, or come from any distribution. In the special case that you know the distribution $B$ comes from, you could use that information to estimate an appropriate choice of $epsilon.alt$. #ref(<cor-gaussian-rank-recovery>, supplement: [Corollary]) gives the case where $B = Z$ is standard Gaussian noise.

#corollary("Rank Recovery with Gaussian Noise")[
Let $Z in bb(R)^(m times n)$ be a Gaussian matrix with standard normal entries $Z_(i j) tilde.op cal(N) (0 \, 1)$ and $sigma_r$ be the largest nonzero singular value of $A in bb(R)^(m times n)$. If $sigma_r > 2 (sqrt(m) + sqrt(n))$, then for $(sqrt(m) + sqrt(n) + t) lt.eq epsilon.alt < sigma_r \/ 2$, $ upright(r a n k)_epsilon.alt (A + Z) = upright(r a n k) (A) $ with high probability at least $1 - 2 exp (- c t^2)$ for some constant $0 < c in bb(R)$.

] <cor-gaussian-rank-recovery>
Extending these results to constrained tensors factorization should be possible, but there presents multiple ways we could approach the problem. In the case of identifying rank for a signal decomposition, we are interested the smallest sum of sources under constraints on the allowed sources. For nonnegative sources, the well studied nonnegative rank @gillis_nonnegative_2020 is most appropriate:

#math.equation(block: true, numbering: "(1)", [ $ upright(r a n k)_(+) (X) = min {R in bb(Z)_(+) mid(bar.v) exists A \, B gt.eq 0 upright(" s.t. ") X [i \, j] = sum_(r = 1)^R A [i \, r] B [r \, j]} . $ ])<eq-nonnegative-rank>

Unfortunately, computing this is NP hard in general @gillis_nonnegative_2020@vavasis_complexity_2010@gillis_geometric_2012. This means extending the $epsilon.alt$-rank to something like an $epsilon.alt$-nonnegative-rank may not be practically computable.

Even in the unconstrained case, we have only kicked the problem down the road and must estimate a suitable $epsilon.alt$ to extract the rank.

There are a number of proposed solutions to this problem that all follow the general principle of Occam's razor of selecting the model with the smallest rank that still achieves a reasonable fit with the data. For automated rank selection, there are many criteria used to balance simplicity with accuracy. The following is a non exhaustive list of methods used in practice in roughly chronological order:

- Akaike @akaike_new_1974@burnham_model_1998 and Bayesian @schwarz_estimating_1978@neath_bayesian_2012 Information Criterion
- Numerical rank @golubRosetakDocumentRank1977 and in combination with polynomial filtering @ubaru_fast_2016
- Matrix perturbation theory @ratsimalahelo_rank_2001
- Consensus clustering and cophenetic correlation coefficient @monti_consensus_2003@brunet_metagenes_2004
- Point of maximum curvature of the approximation error as a function of the rank @satopaa_finding_2011
- Cross validation @austin_tensor_2014
- "SCREE" plot and segmented linear regression @saylor_CharacterizingSedimentSources_2019[Sec. 3.2]
- Minimum description length @fu_model_2019
- Concordance @fogel_rank_2023
- And hypothesis testing @cai_rank_2023.

The number of ways to estimate rank---and lack of consensus on when to use each method---is best summarized by Fu et. al. @fu_model_2019:

#quote(block: true)[
"Even though the selection of rank is very important, in most studies on \[non-negative tensor factorization\] the value of R is simply determined by trial and error or specialists' insights, and there does not exist a good way to determine R automatically."
]

We take an approach similar to Satopaa @satopaa_finding_2011, Saylor @saylor_CharacterizingSedimentSources_2019[Sec. 3.2], and Graham @graham_tracing_2025. We first factorize the tensor at every possible rank (as given by the bounds in #ref(<lem-tensor-rank-bounds>, supplement: [Lemma])), and then look at the function that takes as input a rank $r$, and output the final relative error between the reconstructed decomposition at that rank $X_r^(\*)$ and the input tensor $Y$,

$ f (r) = norm(X_r^(\*) - Y)_F \/ norm(Y)_F . $

We select the "knee" or "elbow" of this function by finding the point of maximum curvature $kappa_f (r)$ with finite differences,

$ hat(R) = arg thin max_r kappa_f (r) := frac(f'' (r), (1 + (f ' (x))^2)^(3 \/ 2)) . $

Since scaling the function $f$ would influence the curvature, we standardize the function by scaling it to the $[0 \, 1]^2$ box. The following is the specific implementation of the rank finding algorithm, and method for calculating the standard curvature.

Since the maximum rank could be very large, we also include an option `online_rank_estimation` which stops looking are larger ranks if the standard curvature has noticeably dropped, implying a local maximum.

```julia
function rank_detect_factorize(Y; online_rank_estimation=false, rank=nothing, model=Tucker1, kwargs...)
    if isnothing(rank)
        # Initialize output and final error lists
        all_outputs = []
        final_rel_errors = Float64[]

        # Make sure RelativeError is part of the stats keyword argument
        kwargs = isempty(kwargs) ? Dict{Symbol,Any}() : Dict{Symbol,Any}(kwargs)
        get!(kwargs, :stats) do # If stats is not given, populate stats with RelativeError
            [Iteration, RelativeError, ObjectiveValue, isnonnegative(Y) ? GradientNNCone : GradientNorm]
        end
        if RelativeError ∉ kwargs[:stats] # If stats was given, make sure RelativeError is in the list stats
            kwargs[:stats] = [RelativeError, kwargs[:stats]...] # not using pushfirst! since kwargs[:stats] could be a Tuple
        end
        kwargs[:model] = model # add the model back into kwargs

        for rank in possible_ranks(Y, model)
            @info "Trying rank=$rank..."

            kwargs[:rank] = rank # add the rank into kwargs

            output = factorize(Y; kwargs...) # safe to call factorize (rather than _factorize) since both factorize and rank_detect_factorize have checks to see if the keyword `rank` is provided
            push!(all_outputs, output)
            _, stats, _ = output

            final_rel_error = stats[end, :RelativeError]
            push!(final_rel_errors, final_rel_error)
            @info "Final relative error = $final_rel_error"

            if (online_rank_estimation == true) && length(final_rel_errors) >= 3 # Need at least 3 points to evaluate curvature
                curvatures = standard_curvature(final_rel_errors)
                if curvatures[end] ≈ maximum(curvatures) # want the last curvature to be significantly smaller than the max
                    continue
                else
                    # we must have curvature[end] < maximum(curvature) so we can now return
                    R = argmax(curvatures)
                    @info "Optimal rank found: $R"
                    return ((all_outputs[R])..., final_rel_errors)
                end
            end
        end

        # Return if online_rank_estimation == false, or a clear rank was not found
        R = argmax(standard_curvature(final_rel_errors))
        @info "Optimal rank found: $R"
        return ((all_outputs[R])..., final_rel_errors)
    else
        return factorize(Y; rank, model, kwargs...)
    end
end
```

```julia
"""
Approximate first derivative with finite elements. Assumes y[i] = y(x_i) are samples with unit spaced inputs x_{i+1} - x_i = 1.
"""
function d_dx(y::AbstractVector{<:Real})
    d = similar(y)
    each_i = eachindex(y)

    # centred estimate
    for i in each_i[begin+1:end-1]
        d[i] = (-y[i-1] + y[i+1])/2
    end

    # three point forward/backward estimate
    i = each_i[begin+1]
    d[begin] = (-3*y[i-1] + 4*y[i] - y[i+1])/2

    i = each_i[end-1]
    d[end] = (y[i-1] - 4*y[i] + 3*y[i+1])/2
    return d
end

function d2_dx2(y::AbstractVector{<:Real})
    d = similar(y)

    for i in eachindex(y)[begin+1:end-1]
        d[i] = y[i-1] - 2*y[i] + y[i+1]
    end

    # Assume the same second derivative at the end points
    d[begin] = d[begin+1]
    d[end] = d[end-1]
    return d
end

function standard_curvature(y::AbstractVector{<:Real}; kwargs...)
    # An interval 0:10 has length(0:10) = 11, but measure 10-0 = 10
    # hence the length(y) - 1
    Δx = 1 / (length(y) - 1)
    y_max = maximum(y)
    dy_dx = d_dx(y; kwargs...) / (Δx * y_max)
    dy2_dx2 = d2_dx2(y; kwargs...) / (Δx^2 * y_max)
    return @. dy2_dx2 / (1 + dy_dx^2)^1.5
end
```

= Computational Techniques
<computational-techniques>
- As stated, algorithm works
- But can be slow, especially for constrained or large problems

As stated, the algorithm described in #ref(<sec-base-algorithm>, supplement: [Section]) works. It will converge to a solution to our optimization problem and factorize the input tensor. It is worth discussing how the algorithm can be modified to improve convergence to maintain quick convergence for large problems, and what sort of architectural methods are used to allow for maximum flexibility, without over engineering the package.

== For Improving Convergence Speed
<for-improving-convergence-speed>
There are a few techniques used to assist convergence. Two ideas that are well studied are discussed in this section. They are 1) breaking up the updates into smaller blocks, and 2) using momentum or acceleration. What is perhaps novel is considering the synergy between these two ideas.

Two more techniques are implemented in BlockTensorFactorization.jl to improve convergence. To the authors knowledge, these are new to tensor factorization, but may or may not be applicable depending on the exact factorization problem or data being studied. For these reasons, these other techniques are discussed separately in #ref(<sec-ppr>, supplement: [Section]) and #ref(<sec-multi-scale>, supplement: [Section]).

=== Sub-block Descent
<sec-sub-block-descent>
- Use smaller blocks, but descent in parallel (sub-blocks don't wait for other sub-blocks)
- Can perform this efficiently with a "matrix step-size"

When using block coordinate descent as in #ref(<sec-base-algorithm>, supplement: [Section]), it is natural to treat each factor as its own block. This requires the fewest blocks while ensuring the objective is still convex with respect to each block. We could just as easily use smaller blocks.

In the case of Tucker decomposition, one modification of the update shown in #ref(<eq-proximal-explicit>, supplement: [Equation]) would be to update each column $a_(n \, r)$, $r = 1 \, dots.h \, R_n$ of the matrix $A_n$ separately. This would be suitable if the constraint that $A_N in cal(C)_n$ can be broken up further into the constraints $a_(n \, r) in cal(C)_(n \, r)$. This is shown in the following update scheme:

#math.equation(block: true, numbering: "(1)", [ $ a_(n \, r)^(t + 1) arrow.l P_(cal(C)_(n \, r)) (a_(n \, r)^t - 1 / L_(n \, r)^t nabla f_(n \, r)^t (a_(n \, r)^t)) \, $ ])<eq-sub-block-update-proper>

where $f_(n \, r)^t (a) = 1 / 2 norm([B \; A_1^(t + 1) \, dots.h \, A_(n - 1)^(t + 1) \, A_(n \, r)^t (a) \, A_(n + 1)^t \, dots.h \, A_N^t] - Y)_F^2$ and

#math.equation(block: true, numbering: "(1)", [ $ A_(n \, r)^t (a) = mat(delim: "[", arrow.t, , arrow.t, arrow.t, arrow.t, , arrow.t; a_(n \, 1)^(t + 1), dots.h.c, a_(n \, r - 1)^(t + 1), a, a_(n \, r + 1)^t, dots.h.c, a_(n \, R_n)^t; arrow.b, , arrow.b, arrow.b, arrow.b, , arrow.b; #none) . $ ])<eq-proper-column-update>

In theory, the block update shown in #ref(<eq-sub-block-update-proper>, supplement: [Equation]) should be a bit more expensive than using the larger blocks on the matrices $A$ shown in #ref(<eq-proximal-explicit>, supplement: [Equation]), since the gradient needs to be recomputed $R_n$ times for each matrix block $n$, rather than only computing the gradient once per block $n$. To get around this, we use the fact that $nabla f_(n \, r)^t (a)$ is the $r$th column from the gradient $nabla f_n^t (A)$ where $f_n^t (A) = 1 / 2 norm([B \; A_1^(t + 1) \, dots.h \, A_(n - 1)^(t + 1) \, A \, A_(n + 1)^t \, dots.h \, A_N^t] - Y)_F^2$. So we can approximate #ref(<eq-sub-block-update-proper>, supplement: [Equation]) by first calculating the gradient $nabla f_n^t$ at

#math.equation(block: true, numbering: "(1)", [ $ hat(A)_n^t = mat(delim: "[", arrow.t, , arrow.t, arrow.t, arrow.t, , arrow.t; a_(n \, 1)^t, dots.h.c, a_(n \, r - 1)^t, a_(n \, r)^t, a_(n \, r + 1)^t, dots.h.c, a_(n \, R_n)^t; arrow.b, , arrow.b, arrow.b, arrow.b, , arrow.b; #none) \, $ ])<eq-merged-column-update>

and then updating each sub-block $r$ according to

#math.equation(block: true, numbering: "(1)", [ $ a_(n \, r)^(t + 1) arrow.l P_(cal(C)_(n \, r)) (a_(n \, r)^t - 1 / L_(n \, r)^t nabla f_n^t (hat(A)_n^t)) . $ ])<eq-sub-block-update-half-merged>

Note the difference between #ref(<eq-proper-column-update>, supplement: [Equation]) and #ref(<eq-merged-column-update>, supplement: [Equation]) is that we don't use the most recent columns $a_(n \, j)$ for $j < r$ in #ref(<eq-merged-column-update>, supplement: [Equation]).

The update given in #ref(<eq-sub-block-update-half-merged>, supplement: [Equation]) can be merged back to an update on the whole block $A_n$

#math.equation(block: true, numbering: "(1)", [ $ A_n^(t + 1) arrow.l P_(cal(C)_n) (A_n^t - nabla f_n^t (A_n^t) (hat(L)_n^t)^(- 1)) $ ])<eq-sub-block-update>

where we have the $R_n times R_n$ diagonal "Lipschitz Matrix"

$ hat(L)_n^t = mat(delim: "[", L_(n \, 1)^t, 0, , 0; 0, L_(n \, 2)^t, , 0; #none, , dots.down, dots.v; 0, 0, dots.h.c, L_(n \, R_n)^t) . $

It is not too hard to show that the Lipschitz $L_(n \, r)^t$ for $nabla f_(n \, r)^t$ is the Euclidean norm of the $r$th column#footnote[We could have used the $r$th row of $tilde(X)_n dot.op_n tilde(X)_n$ since this matrix is symmetric. Since Julia store matrices in column-major order, many operations that perform column-wise are more efficient than their equivalent row-wise operation.] of the matrix $tilde(X)_n dot.op_n tilde(X)_n$ from #ref(<cor-least-squares-matrix>, supplement: [Corollary]),

$ L_(n \, r)^t = norm((tilde(X)_n dot.op_n tilde(X)_n) [: \, r])_2 . $

This leads to the following efficient calculation of the Lipschitz matrix in Julia.

```julia
function diagonal_lipschitz_matrix(T::Tucker, n::Int; kwargs...)
    B = core(T)
    matrices = matrix_factors(T)
    X̃ₙ = tuckerproduct(B, matrices; exclude=n)
    return Diagonal_col_norm(slicewise_dot(X̃ₙ, X̃ₙ; dims=n))
end

Diagonal_col_norm(X) = Diagonal(norm.(eachcol(X)))
```

We can now compare the merged sub-block update #ref(<eq-sub-block-update>, supplement: [Equation]) to the standard projected gradient descent update shown in #ref(<eq-proximal-explicit>, supplement: [Equation]). The difference is that we calculate a "matrix step-size" $hat(L)_n^t in bb(R)^(R_n times R_n)$ rather than a scalar $L_n^t in bb(R)$. In practice, this leads to an improvement in convergence speed for two reasons.

First, computing the matrix $hat(L)_n^t$ often faster than the scalar $L_n^t = norm(tilde(X)_n dot.op_n tilde(X)_n)_(upright("op"))$. The former only requires calculating the Euclidean norm of $R$ vectors for a total cost of $2 R_n^2$ floating point operations (FLOPs), whereas the latter requires the top eigenvalue of $tilde(X)_n dot.op_n tilde(X)_n$. This is usually done with a power method or truncated SVD which can be costlier than the flat rate of $2 R_n^2$ FLOPs.

Secondly, using the matrix $hat(L)_n^t$ means columns where $L_(n \, r)^t$ is small can take larger descent steps. This is because the largest singular value of $tilde(X)_n dot.op_n tilde(X)_n$ is an upper bound on the Euclidean norm of each column: $L_(n \, r)^t lt.eq L_n^t$. Using the scaler Lipschitz $L_n^t$ is equivalent to the diagonal matrix

$ D = mat(delim: "[", L_n^t, 0, , 0; 0, L_n^t, , 0; #none, , dots.down, dots.v; 0, 0, dots.h.c, L_n^t) $

in the merged sub-block update shown in #ref(<eq-sub-block-update>, supplement: [Equation]). So each column of $A_n$ is forced to use the worst case (largest) singular value of $tilde(X)_n dot.op_n tilde(X)_n$. In this way, the matrix $hat(L)_n^t$ acts like a cheap approximate Hessian as if we were doing a quasi-Newton update with step-size $1$.

For completeness, we can perform the same merged sub-block update to update the core $B$. In this case, we obtain the more complicated "Lipschitz tensor" $hat(L)_B^t in bb(R)^((R_1 times dots.h.c times R_N)^2)$ defined by

$ hat(L)_B^t = hat(L)_(B \, 1)^t times.circle dots.h.c times.circle hat(L)_(B \, N)^t $

where each matrix $hat(L)_(B \, n)^t in bb(R)^(R_n times R_n)$ is diagonal with non-zero entries

$ L_(B \, n)^t [r \, r] = norm(((A_n^t)^tack.b A_n^t) [: \, r])_2 . $

The merged sub-block update for the core becomes

#math.equation(block: true, numbering: "(1)", [ $ B^(t + 1) arrow.l P_(cal(C)_B) (B^t - nabla f_0^t (B^t) times_B (hat(L)_B^t)^(- 1)) $ ])<eq-sub-block-update-core>

with the multiplication

#math.equation(block: true, numbering: "(1)", [ $ nabla f_0^t (B^t) times_B (hat(L)_B^t)^(- 1) & = nabla f_0^t (B^t) times.big_n (hat(L)_(B \, n)^t)^(- 1)\
 & = lr(bracket.l.double nabla f_0^t (B^t) \; (hat(L)_(B \, 1)^t)^(- 1) \, dots.h \, (hat(L)_(B \, n)^t)^(- 1) bracket.r.double) . $ ])<eq-tensor-matrices-product>

This should be thought of as normalizing each dimension of the tensor $nabla f_0^t (B^t)$ so that we can take a unit step-size.

TODO use the multi-mode product in stead of defining a new multiplication $times_B$ here. I think I'll have the same tensor product issue as before where the indices.

Putting the core and matrices Lipschitz calculations together gives us the following Julia code. Note we store $hat(L)_B^t$ in factored form as a tuple of diagonal matrices to save space and computation.

TODO Compare to preconditioned decent. See @kunstner_searching_2023@gao_gradient_2024@gao_scalable_2024@qu_optimal_2024

```julia
function make_block_lipschitz(T::AbstractTucker, n::Integer, Y::AbstractArray; objective::L2, kwargs...)
    N = ndims(T)
    if n==0 # the core is the zeroth factor
        function lipschitz_core(X::AbstractTucker; kwargs...)
            return map(A -> Diagonal_col_norm(A'A), matrix_factors(X))
        end # Returns a tuple of diagonal matrices
        return lipschitz_core

    elseif n in 1:N
        function lipschitz_matrix(X::AbstractTucker; kwargs...)
            matrices = matrix_factors(X)
            X̃ₙ = tuckerproduct(core(X), matrices; exclude=n)
            return Diagonal_col_norm(slicewise_dot(X̃ₙ, X̃ₙ; dims=n))
        end
        return lipschitz_matrix

    else
        error("No $(n)th factor in Tucker")
    end
end
```

=== Momentum
<sec-momentum>
- This one is standard
- Use something similar to @xu_BlockCoordinateDescent_2013
- This is compatible with sub-block descent with appropriately defined matrix operations

In practice, we find that extrapolating the iterate based on the prior iterate

#math.equation(block: true, numbering: "(1)", [ $ hat(A)_n^t arrow.l A_n^t + omega_n^t (A_n^t - A_n^(t - 1)) $ ])<eq-extrapolation-ideal>

for some amount of extrapolation $omega_n^t gt.eq 0$ before applying the update #ref(<eq-sub-block-update>, supplement: [Equation]) greatly improves the speed of descent. This can be thought of as a type of momentum where we continue to move in directions that showed a lot of improvement during the last iteration.

Our selection for $omega_n^t$ follows Xu and Yin's method for block coordinate descent @xu_BlockCoordinateDescent_2013, which is itself inspired by Tseng and Yun's coordinate gradient descent method @tseng_coordinate_2009.

Given a parameter#footnote[Usually we pick a number close to $1$. For example, we use the default $delta = 0.9999$.] $delta in \[ 0 \, 1 \)$, we define the momentum parameters and $tau^t$ and $omega_n^t$ according to the following updates

#math.equation(block: true, numbering: "(1)", [ $ tau^0 & = 1\
tau^(t + 1) & arrow.l 1 / 2 (1 + sqrt(1 + 4 (tau^t)^2))\
hat(omega)^t & arrow.l frac(tau^t - 1, tau^(t + 1))\
omega_n^t & arrow.l min (hat(omega)^t \, delta sqrt(hat(L)_n^(t - 1) (hat(L)_n^t)^(- 1))) . $ ])<eq-momentum-parameters>

TODO notation is going to get confusing. We use hat/not hat $L$ for the scaler vs matrix/tensor version. But we use hat/not hat $omega$ for the ideal vs clamped momentum.

What is novel with our approach is that we perform this momentum on the Lipschitz matrices and tensors $hat(L)_n^t$ rather than scaler Lipschitz constant $L_n^t$. In this way, we should interpret the operations shown in #ref(<eq-momentum-parameters>, supplement: [Equation]) as operating element-wise. This also means the momentum parameter $omega_n^t$ is a matrix or tensor and takes the same shape as $hat(L)_n^t$.

In order to perform #ref(<eq-extrapolation-ideal>, supplement: [Equation]), we use the equivalent but more efficient formulation

$ hat(A)_n^t & arrow.l A_n^t (upright("id")_(R_n) + omega_n^t) - A_n^(t - 1) omega_n^t . $

```julia
function (U::MomentumUpdate)(X::T; X_last::T, ω, δ, kwargs...) where T
    n = U.n

    L = U.lipschitz(X; kwargs...)
    L_last = U.lipschitz(X_last; kwargs...)
    ω = min.(ω, δ .* .√(L_last/L))

    A, A_last = factor(X, n), factor(X_last, n)

    A .= U.combine(A, id + ω)
    A .-= U.combine(A_last, ω)
end
```

In the code above, the momentum stores the factor $n$ in acts on, how to compute the Lipschitz constant, matrix, or tensor, and how to combine (multiply) the constant with the factor. In the case of matrix factors $A_n$ in a Tucker decomposition, this is simply right matrix-matrix multiplication. The core factor $B$ uses $times_B$ as described in #ref(<eq-tensor-matrices-product>, supplement: [Equation]).

The parameters of the momentum update are handled separately. This is to treat the momentum update as "apply this update with these parameters". The parameters $tau^t$ and $hat(omega)^t$ are updated by the following function that keeps track of all parameters needed to perform the iteration. In this case, we keep track of what iteration $t$ we are at, the previous iterate, and a few options for the order in which to cycle through and update the blocks.

```julia
update_τ(τ) = 0.5*(1 + sqrt(1 + 4*τ^2))

function initialize_parameters(decomposition, Y, previous; momentum::Bool, random_order, recursive_random_order, kwargs...)
    # parameters for the update step are symbol => value pairs
    # held in a dictionary since we may mutate these, e.g. the step-size
    parameters = Dict{Symbol, Any}()

    # General Looping
    parameters[:iteration] = 0
    parameters[:X_last] = previous[begin] # Last iterate
    parameters[:random_order] = random_order
    parameters[:recursive_random_order] = recursive_random_order

    # Momentum
    if momentum
        parameters[:τ_last] = float(1) # need this field to hold Floats, not Ints
        parameters[:τ] = update_τ(float(1))
        parameters[:ω] = (parameters[:τ_last] - 1) / parameters[:τ]
        parameters[:δ] = kwargs[:δ]
    end

    function updateparameters!(parameters, decomposition, previous)
        parameters[:iteration] += 1
        # parameters[:x_last] = previous[begin]
# This is commented since parameters[:x_last] already points to previous[begin]

        if momentum
            parameters[:τ_last] = parameters[:τ]
            parameters[:τ] = update_τ(parameters[:τ_last])
            parameters[:ω] = (parameters[:τ_last] - 1) / parameters[:τ]
        end
    end

    return parameters, updateparameters!
end
```

=== Empirical Evidence for Sub-Block Descent and Momentum
<empirical-evidence-for-sub-block-descent-and-momentum>
To showcase that the combination of these two tricks can speed up convergence, we will benchmark them by factorizing a random $10 times 10$ tensor (a matrix) with rank $3$. The Julia code is shown below, and the results are shown in #ref(<tbl-subblock-momentum-results>, supplement: [Table]).

```julia
using BenchmarkTools
using BlockTensorFactorization

fact = BlockTensorFactorization.factorize

options = (
    :rank => 3,
    :tolerance => (1, 0.03),
    :converged => (GradientNNCone, RelativeError),
    :δ => 0.9,
)

n_subblock_n_momentum(Y) = fact(Y;
    do_subblock_updates=false,
    momentum=false,
    options...
)

y_subblock_n_momentum(Y) = fact(Y;
    do_subblock_updates=true,
    momentum=false,
    options...
)

n_subblock_y_momentum(Y) = fact(Y;
    do_subblock_updates=false,
    momentum=true,
    options...
)

y_subblock_y_momentum(Y) = fact(Y;
    do_subblock_updates=true,
    momentum=true,
    options...
)

I, J = 10, 10
R = 3

@benchmark n_subblock_n_momentum(Y) setup=(Y=Tucker1((I, J), R))
@benchmark n_subblock_y_momentum(Y) setup=(Y=Tucker1((I, J), R))
@benchmark y_subblock_n_momentum(Y) setup=(Y=Tucker1((I, J), R))
@benchmark y_subblock_y_momentum(Y) setup=(Y=Tucker1((I, J), R))

performance_increase(old, new) = (old - new) / new * 100
```

The code `Tucker1((I, J), R)` produces a random $I times J$ rank-$R$ matrix by generating two matrices $A in bb(R)^(I times R)$ and $B in bb(R)^(R times J)$ with standard normal entries, and multiplies them together.

#figure([
#table(
  columns: (25.33%, 37.33%, 37.33%),
  align: (auto,auto,auto,),
  table.header([], [#strong[No Momentum];], [#strong[Yes Momentum];],),
  table.hline(),
  [#strong[No Sub-Block];], [48.843 ms], [45.738 ms (6.7887% faster)],
  [#strong[Yes Sub-Block];], [27.473 ms (77.785% faster)], [#strong[24.350 ms (100.59% faster)];],
)
], caption: figure.caption(
position: top, 
[
Summary of median times to factorize a random $10 times 10$ rank-3 matrix under different methods. The performance increase is given by the formula $(upright("old") - upright("new")) \/ upright("new")$.
]), 
kind: "quarto-float-tbl", 
supplement: "Table", 
)
<tbl-subblock-momentum-results>


In #ref(<tbl-subblock-momentum-results>, supplement: [Table]), you can see that having both sub-block descent and momentum yields the fastest factorization. Moreover, the performance increase is #emph[more] than simply the performance increases obtained by exclusively sub-block or momentum alone.#footnote[The expected performance increase if sub-block descent and momentum where independence would be $(1 + 0.067887) (1 + 0.77785) = 1.89854$ or only 89.854% faster.] This suggests that there is synergy with these two methods and are best used together.

TODO Repeat this experiment on a less trivial factorization. What I've done above can be done with an SVD in less time. Ideally use a 10 x 10 x 10 Tucker decomposition with rank 2 x 3 x 4.

== For Flexibility
<sec-flexibility>
- there are a number of software engineering techniques used
- these help flexibility for hot swapping and a language for making custom…
  - convergence criterion (and having multiple stopping conditions)
  - probing info during the iterations (stats collected at the end)
  - having multiple constraints and ways to enforce them
  - cyclically or partially randomly or fully randomly update factors
- smart enough to apply these in a reasonable order

There are a number of software engineering techniques used to ensure BlockTensorFactorization.jl is flexible and applicable to a wide range of problems. These enable key algorithmic choices to be hot-swapped and easily compared with each other.

=== Convergence Criteria and Stats
<sec-convergence-criteria>
- Can request info about any factor at each outer iteration
- any subset of stats can be the convergence criteria

Some iterative algorithms produce the exact solution of a problem after a finite number of iterations. Generalized minimal residual method (GMRES) is a good example of this \[TODO cite!\]. Our algorithm, like many others, only converges to the exact solution in the limit as the number of iterations grow. Since we would like a solution in finite time, we must halt the algorithm early.

In finite precision, we can halt the algorithm if we can guarantee the solution is accurate to machine precision. This can often be too strict if convergence is not at a fast enough rate. Furthermore, depending on #emph[why] we are decomposing a tensor, we may want different stats to be within a given a tolerance. BlockTensorFactorization.jl attempts to solve this issue by defining some standard criteria that can be used to halt the algorithm. These are subtypes of the abstract type `AbstractStat` and are listed below.

```julia
# X is the decomposition model e.g. Tucker((B, A1, A2, A2)))
# Y is the input tensor we want to decompose
# norm(X) is the Frobenius norm of X

GradientNorm # norm(∇f(X))
GradientNNCone # sqrt(sum(norm(∇f(Ai)[Ai .> 0 .| ∇f(Ai) .< 0])^2 for Ai in factors(X)))
ObjectiveValue # 1/2 norm(X - Y)^2
ObjectiveRatio # norm(X_last - Y)^2 / norm(X_current - Y)^2
RelativeError # norm(X - Y) / norm(Y)
IterateNormDiff # norm(X_current - X_last)
IterateRelativeDiff # norm(X_current - X_last) / norm(X_last)
```

Most of these are self explanatory, expect perhaps `GradientNNCone`. When performing first order unconstrained optimization, we usually slow down progress when the norm of the gradient is small. In the limit, we expect to converge to a stationary point where the gradient is zero. When performing optimization under nonnegative constraints, and even further restrictions like simplex constraints, is makes more sense to ignore entries of the gradient where `X` is negative and the gradient is positive since those coordinates of `X` will not change after they are projected back to the nonnegative constraint.

TODO add theory of negative gradient is in the normal cone of the constraint?

As many or as few of these stats can be used in the call to `factorize`. Their tolerances can also be set independently of each other. The algorithm stops iterating when at least one of the criteria has a value less than its tolerance. As a fail safe, we also define one more type, `Iteration`, which is always active and halts the algorithm when that many iterations have past.

The `AbstractStat` type can also be subtyped to create custom stats that may be used to probe the iterates and diagnose issues.

```julia
EuclidianStepSize
EuclidianLipschitz
FactorNorms
```

These stats are recorded every iteration in a `DataFrame` and are one of the returns of `factorize`.

Finally, there are two auxiliary stats `PrintStats` and `DisplayDecomposition` which can be used to print all stats or the current decomposition each iteration.

=== Constraints
<sec-constraints>
- one type of update (other than the typical GD update)
- can combine them with composition
  - which is different than projecting onto their intersection!
- Constraint updates combine the constraint with how they are enforced
  - need to go together since there are multiple ways to enforce them e.g.~simplex (see next section)

One of the main motivations for developing BlockTensorFactorization.jl is to solve constrained tensor problems. Other code did not have the expressivity to handle constraints beyond the most common constraints on the factors: nonnegativity and Euclidean normalized columns. To enable flexibility, BlockTensorFactorization.jl defines

```julia
abstract type AbstractConstraint <: Function end
```

which has two interfaces. The first treats the constraint like a function

```julia
(C::AbstractConstraint)(A::AbstractArray)
```

and applies the constraint to an abstract array, and the other interface

```julia
check(C::AbstractConstraint, A::AbstractArray)
```

checks if the array `A` satisfies the constraint `C`. A generic constrain only needs to define these two functions.

```julia
struct GenericConstraint <: AbstractConstraint
    apply::Function # input a AbstractArray -> mutate it so that `check` would return true
    check::Function
end

function (C::GenericConstraint)(D::AbstractArray)
    (C.apply)(D)
end

check(C::GenericConstraint, A::AbstractArray) = (C.check)(A)
```

In general, the function `check` could be defined as the following,

```julia
function check(C, A)
    A_copy = copy(A)
    C(A)
    return A ≈ A_copy
end
```

but there is often an easier way to check if a tensor is constrained than to apply the constraint.

Although our basic block projected gradient descent algorithm #ref(<eq-proximal-explicit>, supplement: [Equation]) relies on Euclidean projections to the relevant constraint set, we want to remain flexible and allow for other maps that move an iterate to the constraint set. So each constraint needs to store more than just the constraint itself (the #emph[what];), but also the map from an iterate to the constraint set (the #emph[how];). When prototyping algorithms, it is worth comparing alternate approaches to see if there is a more efficient method to enforce a given constraint (See #ref(<sec-ppr>, supplement: [Section]) for an example of constraining to a set without Euclidean projections).

The first class of constraints are entry-wise constraints. This is convenient for defining a scalar function that gets applied to all entries of a tensor.

```julia
struct Entrywise <: AbstractConstraint
    apply::Function
    check::Function
end

function (C::Entrywise)(A::AbstractArray)
    A .= (C.apply).(A)
end

check(C::Entrywise, A::AbstractArray) = all((C.check).(A))
```

Some common examples would be a nonnegative constraint, constraining entries to an interval, or constraining entries to be one or zero.

```julia
nonnegative! = Entrywise(x -> max(0, x), x ≥ 0)

IntervalConstraint(a, b) = Entrywise(x -> clamp(x, a, b), x -> a ≤ x ≤ b)

binary! = Entrywise(x > 0.5 ? one(x) : zero(x), x -> x in (0, 1))
```

Another class of constraints are normalizations $cal(C)_(lr(bar.v.double dot.op bar.v.double)_a) = {v mid(bar.v) lr(bar.v.double v bar.v.double)_a = 1}$ for some norm $lr(bar.v.double dot.op bar.v.double)_a$. These can be enforced by a (Euclidean) projection

$ hat(v) in arg thin min_(u in cal(C)_(lr(bar.v.double dot.op bar.v.double)_a)) lr(bar.v.double u - v bar.v.double)_2^2 \, $

or a scaling (assuming $v eq.not 0$)

$ hat(v) arrow.l v / lr(bar.v.double v bar.v.double)_a . $

These operations agree for the Frobenius norm (entry-wise $2$-norm), but are different operations in general. In BlockTensorFactorization.jl, we define these classes as the following.

TODO simplify the following code?

```julia
abstract type AbstractNormalization <: AbstractConstraint end
```

```julia
struct ProjectedNormalization <: AbstractNormalization
    norm::Function # calculate the norm of an AbstractArray
    projection::Function # mutate an AbstractArray so it has norm == 1
    whats_normalized::Function # what part of the array is normalized
                               # e.g. eachrow, or omit for the entire array
end

ProjectedNormalization(norm, projection; whats_normalized=identityslice) =
    ProjectedNormalization(norm, projection, whats_normalized)

function (P::ProjectedNormalization)(A::AbstractArray)
    whats_normalized_A = P.whats_normalized(A)
    (P.projection).(whats_normalized_A)
end

check(P::ProjectedNormalization, A::AbstractArray) =
    all((P.norm).(P.whats_normalized(A)) .≈ 1)
```

```julia
struct ScaledNormalization{T<:Union{Real,AbstractArray{<:Real},Function}} <: AbstractNormalization
    norm::Function # calculate the norm of an AbstractArray
    whats_normalized::Function # what part of the array is normalized
    scale::T # the scale that `whats_normalized` should be normalized to
             # Can be a real, array of reals, or function of the input that
             # that returns a real or array of reals
end

ScaledNormalization(norm;whats_normalized=identityslice,scale=1) =
    ScaledNormalization{typeof(scale)}(norm, whats_normalized, scale)

function (S::ScaledNormalization{T})(A::AbstractArray) where {T<:Union{Real,AbstractArray{<:Real}}}
    whats_normalized_A = S.whats_normalized(A)
    A_norm = (S.norm).(whats_normalized_A) ./ S.scale
    whats_normalized_A ./= A_norm
    return A_norm
end

function (S::ScaledNormalization{T})(A::AbstractArray) where {T<:Function}
    whats_normalized_A = S.whats_normalized(A)
    A_norm = (S.norm).(whats_normalized_A) ./ S.scale(A)
    whats_normalized_A ./= A_norm
    return A_norm
end

check(S::ScaledNormalization{<:Union{Real,AbstractArray{<:Real}}}, A::AbstractArray) =
    all((S.norm).(S.whats_normalized(A)) .≈ S.scale)

check(S::ScaledNormalization{<:Function}, A::AbstractArray) =
    all((S.norm).(S.whats_normalized(A)) .≈ S.scale(A))
```

We can define some common constraints like $L_1$ normalized through a projection

```julia
l1normalize! =
    ProjectedNormalization(l1norm, l1project!)
l1normalize_rows! =
    ProjectedNormalization(l1norm, l1project!; whats_normalized=eachrow)
l1normalize_1slices! =
    ProjectedNormalization(l1norm, l1project!;
        whats_normalized=(x -> eachslice(x; dims=1)))
```

or $L_oo$ normalized by scaling

```julia
linftyscale_cols! = ScaledNormalization(linftynorm; whats_normalized=eachcol)
```

or even normalize the 3-fibres of a third order tensor on average!

```julia
l2scale_average12slices! = ScaledNormalization(l2norm;
    whats_normalized=(x -> eachslice(x; dims=1)),
    scale=(A -> size(A, 2)))
```

Basic linear constrains `AX=B` are also implemented in the following manner. This constraint projects a factor `X` onto the affine space defined by `AX=B` for a linear operator or matrix `A` and bias `B`.

```julia
struct LinearConstraint{T <: Union{Function, AbstractArray}} <: AbstractConstraint
    linear_operator::T
    bias::AbstractArray
end

check(C::LinearConstraint{Function}, X::AbstractArray) = C.linear_operator(X) ≈ C.bias
check(C::LinearConstraint{<:AbstractArray}, X::AbstractArray) = C.linear_operator * X ≈ C.bias

# TODO implement linear constraint given an operator
function (C::LinearConstraint{Function})(X::AbstractArray)
    error("Linear Constraints defined in terms of an operator are not implemented (YET!)")
end

function (C::LinearConstraint{<:AbstractArray})(X::AbstractArray)
    error("Linear Constraints defined in terms of a general array are not implemented (YET!)")
end

function (C::LinearConstraint{<:AbstractMatrix})(X::AbstractArray)
    A = C.linear_operator
    b = C.bias
    X .-= A' * ( (A*A') \ (A*X .- b) ) # Projects X onto the subspace AX=b
end
```

Constraints can also be composed. This is #emph[not] the same as the intersection of constraints. This is only a handy way to apply multiple constraints in series, but there is no clever logic that interprets the constraints and combines them into a single constraint.

```julia
struct ComposedConstraint{T<:AbstractConstraint, U<:AbstractConstraint} <: AbstractConstraint
    outer::T
    inner::U
end

function (C::ComposedConstraint)(A::AbstractArray)
    C.inner(A)
    C.outer(A)
end

check(C::ComposedConstraint, A::AbstractArray) =
    check(C.outer, A) & check(C.inner, A)

Base.:∘(f::AbstractConstraint, g::AbstractConstraint) =
    ComposedConstraint(f, g)
```

This means the following three constraints are all different!

```julia
l1normalize! ∘ nonnegative!
nonnegative! ∘ l1normalize!
simplex!
```

See #ref(<sec-ppr>, supplement: [Section]) for a discussion on why it may be advantages to use one of these constraints over the other.

=== `BlockUpdate` Language
<blockupdate-language>
- construct the updates as a list of updates
- very functional programming
- can apply them in sequence or in a random order (or partially random)

To put all these pieces together, we define an abstract type that can be subtyped for the various types of update. This will include gradient descent steps, momentum updates, and constraint enforcing updates.

```julia
abstract type AbstractUpdate <: Function end
```

A generic update is a function that can take some keyword arguments and mutate an iterate.

```julia
struct GenericUpdate <: AbstractUpdate
    f::Function
end

(U::GenericUpdate)(x; kwargs...) = U.f(x; kwargs...)
```

The first types of updates are gradient descent updates.

```julia
abstract type AbstractGradientDescent <: AbstractUpdate end
```

This includes the regular gradient descent update, and also the sub-block descent updates.

```julia
struct GradientDescent <: AbstractGradientDescent
    n::Integer
    gradient::Function
    step::AbstractStep
end

function (U::GradientDescent)(x; x_last, kwargs...)
    n = U.n
    if checkfrozen(x, n)
        return x
    end
    grad = U.gradient(x; kwargs...)
    # Note we pass a function for grad_last (lazy) so that we only compute it if needed for the step
    s = U.step(x; n, x_last, grad, grad_last=(x -> U.gradient(x; kwargs...)), kwargs...)
    a = factor(x, n)
    @. a -= s*grad
end
```

The only addition to `BlockGradientDescent` is a function that combines the step with the gradient.

```julia
struct BlockGradientDescent <: AbstractGradientDescent
    n::Integer
    gradient::Function
    step::AbstractStep
    combine::Function # takes a step (number, matrix, or tensor) and combines it with a gradient
end

function (U::BlockGradientDescent)(x; x_last, kwargs...)
    n = U.n
    if checkfrozen(x, n)
        return x
    end
    grad = U.gradient(x; kwargs...)
    # Note we pass a function for grad_last (lazy) so that we only compute it if needed for the step
    s = U.step(x; n, x_last, grad, grad_last=(x -> U.gradient(x; kwargs...)), kwargs...)
    a = factor(x, n)
    a .-= U.combine(grad, s)
end
```

A separate step type is defined to allow for different types of steps like constant step, Lipschitz, and spectral projected gradient (SPG).

```julia
abstract type AbstractStep <: Function end
```

```julia
struct LipschitzStep <: AbstractStep
    lipschitz::Function
end

function (step::LipschitzStep)(x; kwargs...)
    L = step.lipschitz(x)
    return L^(-1) # allow for Lipschitz to be a diagonal matrix
end

function (step::LipschitzStep)(x::Tucker; kwargs...)
    L = step.lipschitz(x)
    if typeof(L) <: Tuple # Currently the only case is when we are updating the core of a Tucker factorization
                          # Using this condition as a way to tell if it is the core we are calculating the constant for
        return map(X -> X^(-1), L)
    else
        return L^(-1) # allow for Lipschitz to be a diagonal matrix
    end
end
```

```julia
struct ConstantStep <: AbstractStep
    stepsize::Real
end

(step::ConstantStep)(x; kwargs...) = step.stepsize
```

```julia
struct SPGStep <: AbstractStep
    min::Real
    max::Real
end

SPGStep(;min=1e-10, max=1e10) = SPGStep(min, max)

# Convert an input of the full decomposition, to a calculation on the nth factor
(step::SPGStep)(x::T; n, x_last::T, grad_last::Function, kwargs...) where {T <: AbstractDecomposition} =
    step(factor(x,n); x_last=factor(x_last,n), grad_last=grad_last(x_last), kwargs...)

function (step::SPGStep)(x; grad, x_last, grad_last, stepmin=step.min, stepmax=step.max, kwargs...)
    s = x - x_last
    y = grad - grad_last
    sy = (s ⋅ y)
    if sy <=0
        return stepmax
    else
        suggested_step = (s ⋅ s) / sy
        return clamp(suggested_step, stepmin, stepmax)
    end
end
```

The gradient and Lipschitz stepsize is calculated by a separate function that gets make on initialization of the `factorization` algorithm (See #ref(<sec-gradient-computation>, supplement: [Section]) and #ref(<sec-lipschitz-computation>, supplement: [Section])). This de-couples applying the gradient descent (the #emph[what];), and the computation of the gradient (the #emph[how];) so that the gradient can be calculated manually (if an efficient method is known), or with automatic differentiation. This also allows other updates to use the same computation code. For example, momentum updates also use the same Lipschitz calculation function. Momentum updates also use the same `combine` function as the sub-block gradient descent updates.

```julia
struct MomentumUpdate <: AbstractUpdate
    n::Integer
    lipschitz::Function
    combine::Function # How to combine the momentum variable `ω` with a factor `a`
end

MomentumUpdate(n, lipschitz) = MomentumUpdate(n, lipschitz, (ω, a) -> ω * a)

function MomentumUpdate(GD::AbstractGradientDescent)
    n, step = GD.n, GD.step
    @assert typeof(step) <: LipschitzStep

    return MomentumUpdate(n, step.lipschitz)
end

function MomentumUpdate(GD::BlockGradientDescent)
    n, step, combine = GD.n, GD.step, GD.combine
    @assert typeof(step) <: LipschitzStep

    return MomentumUpdate(n, step.lipschitz, combine)
end
```

Constraints also get their own abstract subtype of updates.

```julia
abstract type ConstraintUpdate <: AbstractUpdate end
```

We define a constructor for applying a constraint to a particular factor $n$. This turns an `AbstractConstraint` into a `ConstraintUpdate`.

```julia
ConstraintUpdate(n, constraint::GenericConstraint; kwargs...) =
    GenericConstraintUpdate(n, constraint)
ConstraintUpdate(n, constraint::ProjectedNormalization; kwargs...) =
    Projection(n, constraint)
ConstraintUpdate(n, constraint::Entrywise; kwargs...) =
    Projection(n, constraint)

function ConstraintUpdate(n, constraint::ScaledNormalization; skip_rescale=false, whats_rescaled=missing, kwargs...)
    if skip_rescale
        ismissing(whats_rescaled) ||
        isnothing(whats_rescaled) ||
        @warn "skip_rescale=true but whats_rescaled=$whats_rescaled was given. Overriding to whats_rescaled=nothing"
        return Rescale(n, constraint, nothing)
    else
        return Rescale(n, constraint, whats_rescaled)
    end
end
```

A generic constraint update extracts the factor it needs to constrain, and applies the constraint to that factor.

```julia
struct GenericConstraintUpdate <: ConstraintUpdate
    n::Integer
    constraint::GenericConstraint
end

check(U::GenericConstraintUpdate, D::AbstractDecomposition) = check(U.constraint, factor(D, U.n))

function (U::GenericConstraintUpdate)(x::T; kwargs...) where T
    n = U.n
    A = factor(x, n)
    U.constraint(A)
    check(U, A) ||
        error("Something went wrong with GenericConstraintUpdate: $GenericConstraintUpdate")
end
```

Projections follow this pattern closely.

```julia
struct Projection <: ConstraintUpdate
    n::Integer
    proj::Union{ProjectedNormalization, Entrywise}
end

check(P::Projection, D::AbstractDecomposition) = check(P.proj, factor(D, P.n))

function (U::Projection)(x::T; kwargs...) where T
    n = U.n
    U.proj(factor(x, n))
end
```

A more involved type of constraint update is a rescaled update. This uses a scaled normalization, but moves the weight of the factor to other factors.

For example, if we have three matrix factors $A \, B \, C$ in a CP decomposition where we want the sum of the entries in $A$ to sum to $1$, we can divide $A$ by its sum, and multiple factor $B$ by this amount. That way the recombined tensor $⟦A \, B \, C⟧$ remains unchanged, but now $A$ satisfies the desired constraint. We could instead multiply both $B$ and $C$ by the square root of the sum to achieve a similar outcome. When it is not specified what is rescaled (`whats_rescaled = missing`), we assume we should multiply every other factor by the geometric mean of the scaling. If we just want to scale the factor but skip any rescaling, we can use `whats_rescaled = nothing`.

```julia
struct Rescale{T<:Union{Nothing,Missing,Function}} <: ConstraintUpdate
    n::Integer
    scale::ScaledNormalization
    whats_rescaled::T
end

check(S::Rescale, D::AbstractDecomposition) = check(S.scale, factor(D, S.n))

function (U::Rescale{<:Function})(x; kwargs...)
    Fn_scale = U.scale(factor(x, U.n))
    to_scale = U.whats_rescaled(x)
    to_scale .*= Fn_scale
end

(U::Rescale{Nothing})(x; kwargs...) =
    U.scale(factor(x, U.n))

function (U::Rescale{Missing})(x; skip_rescale=false, kwargs...)
    Fn_scale = U.scale(factor(x, U.n))
    x_factors = factors(x)
    N = length(x_factors) - 1

    # Nothing to rescale, so return here
    if N == 0 || skip_rescale
        return nothing
    end

    # Assume we want to evenly rescale all other factors by the Nth root of Fn_scale
    scale = geomean(Fn_scale)^(1/N)
    for (i, A) in zip(eachfactorindex(x), x_factors)
        # skip over the factor we just updated
        if i == U.n
            continue
        end
        A .*= scale
    end
end
```

See #ref(<sec-ppr>, supplement: [Section]) for a more in-depth discussion on when it may be beneficial to use this type of constraint update over a simple projection.

We would like to combine all these updates to execute them in serial. We use a `BlockedUpdate` type to do this. This should be thought of as a list of `AbstractUpdates` that get applied one after another.

```julia
struct BlockedUpdate <: AbstractUpdate
    updates::Vector{AbstractUpdate}
end
```

#block[
#callout(
body: 
[
We need the updates to be exactly something of the form `AbstractUpdate[]` since we want to push any type of `AbstractUpdate`s such as a `MomentumUpdate` or another `BlockedUpdate`, even if not already present. This means it cannot be `Vector{<:AbstractUpdate}` since a `BlockedUpdate` constructed with only `GradientDescent` would give a `GradientDescent[]` vector and we couldn't push a `MomentumUpdate`. And it cannot be `AbstractVector{AbstractUpdate}` since we may not be able to `insert!` or `push!` into other `AbstractVectors` like `Views`.

]
, 
title: 
[
Technical Julia Note
]
, 
background_color: 
rgb("#dae6fb")
, 
icon_color: 
rgb("#0758E5")
, 
icon: 
"❕"
, 
body_background_color: 
white
)
]
We forward many standard methods so that `BlockedUpdate`s can behave like usual Julia vectors.

TODO should I actually show all these details?

```julia
Base.getindex(U::BlockedUpdate, i::Int) = getindex(updates(U), i)
Base.getindex(U::BlockedUpdate, I::Vararg{Int}) = getindex(updates(U), I...)
Base.getindex(U::BlockedUpdate, I) = getindex(updates(U), I) # catch all
Base.firstindex(U::BlockedUpdate) = firstindex(updates(U))
Base.lastindex(U::BlockedUpdate) = lastindex(updates(U))
Base.keys(U::BlockedUpdate) = keys(updates(U))
Base.length(U::BlockedUpdate) = length(updates(U))
Base.iterate(U::BlockedUpdate, state=1) = state > length(U) ? nothing : (U[state], state+1)
Base.filter(f, U::BlockedUpdate) = BlockedUpdate(filter(f, updates(U)))
```

We define how `BlockedUpdates` get applied to a tensor with the following function.

```julia
function (U::BlockedUpdate)(x::T; recursive_random_order::Bool=false, random_order::Bool=recursive_random_order, kwargs...) where T
    U_updates = updates(U)
    if random_order
        order = shuffle(eachindex(U_updates))
        U_updates = U_updates[order]
    end

    for update! in U_updates
        update!(x; recursive_random_order, kwargs...)
        # note random_order does not get passed down
    end
end
```

The default order the blocks are updated is cyclically through each factor of the decomposition `D::AbstractDecomposition`, in the order of `factors(D)`. For `AbstractTucker` decompositions like Tucker, Tucker-1, and CP, this means starting with the core, followed by the matrix factor for the first dimension, second dimension, and so on.

As an example, this would be the default order of updates for nonnegative CP decomposition on an order 3 tensor.

```julia
BlockedUpdate(
    MomentumUpdate(1, lipschitz)
    GradientStep(1, gradient, LipschitzStep)
    Projection(1, Entrywise(ReLU, isnonnegative))
    MomentumUpdate(2, lipschitz)
    GradientStep(2, gradient, LipschitzStep)
    Projection(2, Entrywise(ReLU, isnonnegative))
    MomentumUpdate(3, lipschitz)
    GradientStep(3, gradient, LipschitzStep)
    Projection(3, Entrywise(ReLU, isnonnegative))
)
```

The order of updates can be randomized with the `random_order` keyword.

```julia
X, stats, kwargs = factorize(Y; random_order=true)
```

By default, this will keep momentum steps, gradient steps, and constraint steps for each factor together as a block, in this order.

A possible order of updates could be the following. Note that the updates for each factor are grouped together, but each factor is updated in a random order.

```julia
BlockedUpdate(
    BlockedUpdate(
        MomentumUpdate(2, lipschitz)
        GradientStep(2, gradient, LipschitzStep)
        Projection(2, Entrywise(ReLU, isnonnegative))
    )
    BlockedUpdate(
        MomentumUpdate(1, lipschitz)
        GradientStep(1, gradient, LipschitzStep)
        Projection(1, Entrywise(ReLU, isnonnegative))
    )
    BlockedUpdate(
        MomentumUpdate(3, lipschitz)
        GradientStep(3, gradient, LipschitzStep)
        Projection(3, Entrywise(ReLU, isnonnegative))
    )
)
```

For more randomization, use the `recursive_random_order` keyword which will also randomize the order in which the momentum steps, gradient steps, and constraint steps are performed.

```julia
X, stats, kwargs = factorize(Y; recursive_random_order=true)
```

A possible order of updates could now be the following. The updates for each factor are still grouped together, but the updates within each block appear in a random order.

```julia
BlockedUpdate(
    BlockedUpdate(
        Projection(2, Entrywise(ReLU, isnonnegative))
        MomentumUpdate(2, lipschitz)
        GradientStep(2, gradient, LipschitzStep)
    )
    BlockedUpdate(
        MomentumUpdate(1, lipschitz)
        Projection(1, Entrywise(ReLU, isnonnegative))
        GradientStep(1, gradient, LipschitzStep)
    )
    BlockedUpdate(
        GradientStep(3, gradient, LipschitzStep)
        Projection(3, Entrywise(ReLU, isnonnegative))
        MomentumUpdate(3, lipschitz)
    )
)
```

The opposite of this would be to keep the outer order of blocks as given, but randomize the order which the updates for each factor gets applied, use the following code.

```julia
X, stats, kwargs = factorize(Y; recursive_random_order=true, random_order=false, group_by_factor=true)
```

A possible order of updates could now be the following. Note the order of factors is preserved (1, 2, 3) but the inner `BlockedUpdate`s have a random order.

```julia
BlockedUpdate(
    BlockedUpdate(
        Projection(1, Entrywise(ReLU, isnonnegative))
        MomentumUpdate(1, lipschitz)
        GradientStep(1, gradient, LipschitzStep)
    )
    BlockedUpdate(
        MomentumUpdate(2, lipschitz)
        Projection(2, Entrywise(ReLU, isnonnegative))
        GradientStep(2, gradient, LipschitzStep)
    )
    BlockedUpdate(
        GradientStep(3, gradient, LipschitzStep)
        MomentumUpdate(3, lipschitz)
        Projection(3, Entrywise(ReLU, isnonnegative))
    )
)
```

Note all the previously mentioned options still keeps the various updates for each factor together. For full randomization, use the following code.

```julia
X, stats, kwargs = factorize(Y; recursive_random_order=true, group_by_factor=false)
```

A possible order of updates could now be the following. Note that every update can appear anywhere in the order.

```julia
BlockedUpdate(
    Projection(3, Entrywise(ReLU, isnonnegative))
    MomentumUpdate(2, lipschitz)
    GradientStep(2, gradient, LipschitzStep)
    MomentumUpdate(1, lipschitz)
    GradientStep(1, gradient, LipschitzStep)
    Projection(2, Entrywise(ReLU, isnonnegative))
    MomentumUpdate(3, lipschitz)
    MomentumUpdate(2, lipschitz)
    Projection(1, Entrywise(ReLU, isnonnegative))
    GradientStep(3, gradient, LipschitzStep)
)
```

The complete behaviour is summarized in #ref(<tbl-blockupdate-randomization>, supplement: [Table]).

We also use `BlockedUpdate` to handle a composition of constraints.

```julia
ConstraintUpdate(n, constraint::ComposedConstraint; kwargs...) =
    BlockedUpdate(ConstraintUpdate(n, constraint.inner; kwargs...),
                  ConstraintUpdate(n, constraint.outer; kwargs...))
end
```

The `BlockUpdate` language becomes especially helpful for automatically inserting additional updates as they are requested. For example, if we want to add momentum to a list of updates, we can call the following function.

```julia
function add_momentum!(U::BlockedUpdate)
    # Find all the GradientDescent updates
    U_updates = updates(U)
    indexes = findall(u -> typeof(u) <: AbstractGradientDescent, U_updates)

    # insert MomentumUpdates before each GradientDescent
    # do this in reverse order so "i" correctly indexes a GradientDescent
    # as we mutate updates
    for i in reverse(indexes)
        insert!(U_updates, i, MomentumUpdate(U_updates[i]))
    end
end
```

We can also interlace two lists of updates to put updates on the same factor next to each other, or group all updates by the factor they act on.

```julia
function smart_interlace!(U::BlockedUpdate, other_updates)
    for V in other_updates
        smart_insert!(U::BlockedUpdate, V::AbstractUpdate)
    end
end

smart_interlace!(U::BlockedUpdate, V::BlockedUpdate) = smart_interlace!(U::BlockedUpdate, updates(V))

function smart_insert!(U::BlockedUpdate, V::AbstractUpdate)
    U_updates = updates(U)
    i = findlast(u -> u.n == V.n, U_updates)

    # insert the other update immediately after
    # or if there is no update, push it to the end
    isnothing(i) ? push!(U_updates, V) : insert!(U_updates, i+1, V)
end
```

```julia
function group_by_factor(blockedupdate::BlockedUpdate)
    factor_labels = unique(getproperty(U, :n) for U in blockedupdate)
    updates_by_factor = [filter(U -> U.n == n, blockedupdate) for n in factor_labels]
    return BlockedUpdate(updates_by_factor)
end
```

= Rescaling to Constrain Tensor Factorization
<sec-ppr>
- for bounded linear constraints
  - first project
  - then rescale to enforce linear constraints
- faster to execute then a projection
- often does not loose progress because of the rescaling (decomposition dependent)

Constraints on factors in a tensor decomposition can arise naturally when modeling physical problems. A common class of constraints is normalizations which restrict a factor or slices of a factor to have unit norm. These are sometimes intersected with interval constraints such as requiring entries to be nonnegative.

With constraints being defined in a flexible manner (see #ref(<sec-constraints>, supplement: [Section])), we decided to test conventional wisdom that Euclidean projections are the right kind of map to use when enforcing a constraint. A constraint that came up in recent applications of tensor decomposition to geology @graham_tracing_2025 was enforcing the $1$st mode slices of a tensor (e.g.~rows in a matrix) or $3$rd mode fibres of a third order tensor to lie in their respective simplex.

For example, the demixing of $R$ probability densities $b_1 \, dots.h \, b_R$ from $I$ mixtures $y_1 \, dots.h \, y_I$ for $I > R$ can be accomplished with a nonnegative matrix factorization (cite). We have the system of equations

$ med y_1 & = a_11 med b_1 + a_12 med b_2 + dots.h + a_(1 R) med b_R\
med y_2 & = a_21 med b_1 + a_22 med b_2 + dots.h + a_(2 R) med b_R\
 & dots.v\
med y_I & = a_(I 1) med b_1 + a_(I 2) med b_2 + dots.h + a_(I R) med b_R\
 $

with unknown mixing coefficients $a_(i \, r)$ and densities $b_r$. If we can discretized the mixtures $y_i$, we can rewrite this system as a rank $R$ factorization of the matrix $Y$ where $Y [i \, :] = y_i$ and

$ mat(delim: "[", arrow.l, med y_1^tack.b, arrow.r; arrow.l, med y_2^tack.b, arrow.r; arrow.l, med dots.v, arrow.r; arrow.l, med y_I^tack.b, arrow.r) & = mat(delim: "[", a_11, a_12, dots.h, a_(1 R); a_21, a_22, dots.h, a_(2 R); dots.v, , , dots.v; a_(I 1), a_(I 2), dots.h, a_(I R); #none) mat(delim: "[", arrow.l, med b_1^tack.b, arrow.r; arrow.l, med b_2^tack.b, arrow.r; med, dots.v, med; arrow.l, med b_R^tack.b, arrow.r) $

$ Y = A B \, $

for matrices $A$ and $B$.

For the factorization to remain interpretable, we need to ensure each row $b_r$ of $B$ is a density. This means we would like to constrain each row $B [r \, :] = b_r$ to the simplex#footnote[This assumes the entries $b_r [j]$ in the discretization of the density $b_r$ represent probabilities or areas under some continuous 1D density function, and not the sample values of the density $b_r (x_j)$. One possible discretization is to take $J$ sample points $x_j$ of a grid on the interval $[x_0 \, x_J]$ where $b_r$ is supported, and define the entries of the discretization to be $B [r \, j] = b_r [j] = b_r (x_j) (x_j - x_(j - 1))$. For normalized densities $integral_(x_0)^(x_J) b_r (x) d x = 1$, the sum of the entries $sum_(j = 1)^J b_r [j] approx 1$ when large enough number of samples $J$ are taken.]

$ b_r in Delta_J = {v in bb(R)_(+)^J mid(bar.v) sum_(j = 1)^J v [j] = 1} \, $

which we can write as constraining the matrix $B$ to the simplex

$ B in Delta_J^R = {B in bb(R)_(+)^(R times J) mid(bar.v) forall r in [R] \, thin sum_(j = 1)^J B [r \, j] = 1} . $

Given the rows of $B$ represent densities, to ensure the rows of the reconstructed matrix $hat(Y) = A B$ are still densities, we need the mixing coefficients to be nonnegative $a_(i r) gt.eq 0$ and rows to sum to one $sum_r a_(i r) = 1$. This constrains the matrix $A$ to the simplex

$ A in Delta_R^I = {A in bb(R)_(+)^(I times R) mid(bar.v) forall i in [I] \, thin sum_(r = 1)^R A [i \, r] = 1} . $

The question we investigate in this section is the following: how can we best constrain the factors $A$ and $B$ to their respective simplexes, while performing block gradient decent to minimize the least squared error $1 / 2 norm(A B - Y)_F^2$?

== The two approaches for simplex constraints
<the-two-approaches-for-simplex-constraints>
To constrain a vector $v in bb(R)^J$ to the simplex

$ Delta_J = {v in bb(R)_(+)^J mid(bar.v) sum_(j = 1)^J v [j] = 1} \, $

we could apply a Euclidean projection

$ v arrow.l arg thin min_(u in Delta_J) lr(bar.v.double u - v bar.v.double)_2^2 \, $

or a generalized Kullback-Leibler (KL) divergence projection

#math.equation(block: true, numbering: "(1)", [ $ v arrow.l arg thin min_(u in Delta_J) sum_j u [j] log (frac(u [j], v [j])) - u [j] + v [j] $ ])<eq-kl-projection>

among other reasonable maps onto $Delta_J$.

The Euclidean simplex projection can be done with the following implementation of Chen and Ye's algorithm @chen_projection_2011. The essence of the algorithm is to efficiently compute the special $t in bb(R)$ so that

#math.equation(block: true, numbering: "(1)", [ $ v arrow.l max (0 \, v - t bb(1)) in Delta_J . $ ])<eq-simplex-projection>

The $max (0 \, x)$ function should be understood as operating entrywise on $x$. In BlockTensorFactorization, we use the helper `ReLU(x) = max(0, x)` for this function to assist with broadcasting.

```julia
function projsplx(v)
    J = length(v)

    if J==1 # quick exit for trivial length-1 "vectors" (i.e. scalars)
        return [one(eltype(v))]
    end

    v_sorted = sort(v[:]) # Vectorize/extract input and sort all entries
    j = J - 1
    t = 0 # need to ensure t has scope outside the while loop
    while true
        t = (sum(@view v_sorted[j+1:end]) - 1) / (J-j)
        if t >= v_sorted[j]
            break
        else
            j -= 1
        end

        if j >= 1
            continue
        else # j == 0
            t = (sum(v_sorted) - 1) / J
            break
        end
    end
    return ReLU.(v .- t)
end
```

This is turned into an operation that can mutated `v` with the following definition,

```julia
function projsplx!(y)
    y .= projsplx(y)
end
```

and can be turned into a `ProjectedNormalization` (see #ref(<sec-constraints>, supplement: [Section])) with the following code.

```julia
simplex! = ProjectedNormalization(isnonnegative_sumtoone, projsplx!)
isnonnegative_sumtoone(x) = all(isnonnegative, x) && sum(x) ≈ 1
```

The generalized Kullback-Leibler divergence projection as stated in #ref(<eq-kl-projection>, supplement: [Equation]) is only well-defined when $v [j] > 0$ for all $j in [J]$. In this case the solution is given by

$ v arrow.l frac(v, sum_j v [j]) $

which is well described by Ducellier et. al. @ducellier_uncertainty_2024[Sec. 2.1].

To extend the applicability of this map to any $v$ (when there is at least one positive entry $v [j] > 0$), we can first (Euclidean) project onto the nonnegative orthant $bb(R)_(+)^J$,

$ v arrow.l arg thin min_(u in bb(R)_(+)^J) lr(bar.v.double u - v bar.v.double)_2^2 = max (0 \, v) \, $

and then apply the divergence projection.#footnote[In the unfortunate case where every entry of $v$ is nonpositive, we can fallback to the Euclidean simplex projection.] All together, this looks like #math.equation(block: true, numbering: "(1)", [ $ v arrow.l frac(max (0 \, v), sum_j max (0 \, v [j])) . $ ])<eq-nnpr>

We will refer to #ref(<eq-nnpr>, supplement: [Equation]) as nonnegative projection and rescaling (NNPR). NNPR has the following implementation in BlockTensorFactorization.jl.

```julia
l1scale! ∘ nonnegative!
```

We define the two constraints as the following, using the constraint language from #ref(<sec-constraints>, supplement: [Section]).

```julia
nonnegative! = Entrywise(ReLU, isnonnegative)
l1scale! = ScaledNormalization(l1norm)
l1norm(x) = mapreduce(abs, +, x)
```

== The Rescaling Trick for Matrix Factorization
<sec-matrix-rescaling>
- Explain that we can move the weight from one matrix to another

Comparing the two methods of constraining a vector to the simplex 1) by Euclidean projection (#ref(<eq-simplex-projection>, supplement: [Equation])) or 2) nonnegative projection and rescaling (NNPR, #ref(<eq-nnpr>, supplement: [Equation])), the latter offers a few advantages. NNPR is cheaper and conceptually easier to compute. Another advantage of NNPR to tensor factorizations, is its ability to constrain a factor without loosing progress while performing gradient descent.

For example, consider the low rank factorization problem of finding matrices $A \, B$ such that $Y = A B$ where you would like the sum of entries in $B$ to be one

#math.equation(block: true, numbering: "(1)", [ $ min_(A in bb(R)^(I times R) \, B in bb(R)^(R times J)) 1 / 2 norm(A B - Y)_F^2 quad upright("s.t.") quad B in Delta_(R J) = {B in bb(R)_(+)^(R times J) mid(bar.v) sum_(r \, j) B [r \, j] = 1} . $ ])<eq-B-full-simplex-problem>

The basic alternating projected gradient descent algorithm using a Euclidean projection would be

$ A & arrow.l A - 1 / L_A nabla_A f (A \, B)\
B & arrow.l E P_(Delta_(R J)) (B - 1 / L_B nabla_B f (A \, B)) $

where $f (A \, B) = 1 / 2 norm(A B - Y)_F^2$ and $E P_(Delta_(R J))$ is the Euclidean projection onto the simplex $Delta_(R J)$. In the event the updated value for $B$,

$ B - 1 / L_B nabla_B f (A \, B) := hat(B) in bb(R)_(+)^(R times J) $

is already nonnegative, the objective $f$ at the new point $(A \, E P_(Delta_(R J)) (hat(B)))$ could be bigger or smaller than the objective before the Euclidean projection $f (A \, hat(B))$.

If we use the nonnegative projection and rescaling $upright("NNPR")_(Delta_(R J))$ instead of the Euclidean projection $E P_(Delta_(R J))$ when $hat(B)$ is already nonnegative, then $ upright("NNPR")_(Delta_(R J)) (hat(B)) = frac(1, sum_(r j) B [r \, j]) hat(B) := c^(- 1) hat(B) . $

This means the objective value $f$ at the point $(c^(- 1) A \, c hat(B))$ will be the same as the objective value before the KL divergence projection

$ f (c A \, upright("NNPR")_(Delta_(R J)) (hat(B))) = f (c A \, c^(- 1) hat(B)) = 1 / 2 norm(A c c^(- 1) B - Y)_F^2 = 1 / 2 norm(A B - Y)_F^2 = f (A \, hat(B)) . $

This suggests the following update may be a useful alternative to the standard projected gradient descent

$ A & arrow.l A - 1 / L_A nabla_A f (A \, B)\
B & arrow.l B - 1 / L_B nabla_B f (A \, B)\
c & arrow.l sum_(r \, j) B [r \, j]\
A & arrow.l c A\
B & arrow.l c^(- 1) B . $

Of course, it is possible that $hat(B) = B - 1 / L_B nabla_B f (A \, B)$ has negative entries. So we use both the nonnegative projection and rescaling part of NNPR in the alternating gradient descent with rescaling update

#math.equation(block: true, numbering: "(1)", [ $ A & arrow.l A - 1 / L_A nabla_A f (A \, B)\
B & arrow.l max (0 \, B - 1 / L_B nabla_B f (A \, B))\
c & arrow.l sum_(r \, j) B [r \, j]\
A & arrow.l c A\
B & arrow.l c^(- 1) B . $ ])<eq-nnpr-gd>

This algorithm can be called in BlockTensorFactorization.jl with the following code.

```julia
options = (
    model=Tucker1,
    constraints=[l1scale! ∘ nonnegative!, noconstraint],
)

decomposition, stats, kwargs = factorize(Y; options...);
```

== Generalizing the Matrix Rescaling Trick
<generalizing-the-matrix-rescaling-trick>
The rescaling trick discussed in #ref(<sec-matrix-rescaling>, supplement: [Section]) applies more generally to other tensor factorization, other simplex-type constraints, and other `ScaledNormalization`s.

=== Simplex-type constraints
<simplex-type-constraints>
Instead of the matrix factorization problem where $B in Delta_(R J)$ is constrained to the full simplex (#ref(<eq-B-full-simplex-problem>, supplement: [Equation])), we can apply the rescaling trick the problem where the rows of $B$ are constrained to the simplex

#math.equation(block: true, numbering: "(1)", [ $ min_(A in bb(R)^(I times R)\
B in bb(R)^(R times J)) 1 / 2 norm(A B - Y)_F^2 quad upright("s.t.") quad B in Delta_J^R = {B in bb(R)_(+)^(R times J) mid(bar.v) forall r in [R] \, thin sum_(j = 1)^J B [r \, j] = 1} . $ ])<eq-B-row-simplex-problem>

This could make sense in applications where rows of $B$ represent probability densities such as the demixing problem discussed at the start of this section (#ref(<sec-ppr>, supplement: [Section])).

We can adjust the alternating gradient descent update with NNPR (#ref(<eq-nnpr-gd>, supplement: [Equation])) to the following update #math.equation(block: true, numbering: "(1)", [ $ A & arrow.l A - 1 / L_A nabla_A f (A \, B)\
B & arrow.l max (0 \, B - 1 / L_B nabla_B f (A \, B))\
C [r \, r] & arrow.l sum_j B [r \, j] quad upright("(") C in bb(R)^(R times R) upright(" is diagonal)")\
A & arrow.l A C\
B & arrow.l C^(- 1) B . $ ])<eq-nnpr-gd-B-rows>

It is clear that the objective value would be maintained for any invertible matrix $C$

$ f (A C \, C^(- 1) B) = 1 / 2 norm(A C C^(- 1) B - Y)_F^2 = 1 / 2 norm(A B - Y)_F^2 = f (A \, B) . $

This algorithm can be called in BlockTensorFactorization.jl with the following code.

```julia
options = (
    model=Tucker1,
    constraints=[l1scale_rows! ∘ nonnegative!, noconstraint],
)

decomposition, stats, kwargs = factorize(Y; options...);
```

#block[
#callout(
body: 
[
This trick would also work if we wanted the columns of $A$ normalized to the simplex. But it does #emph[not] work when we would like each column of $B$ to be constrained to the simplex. The normalizing matrix $C$ would have to be multiplied to the right of $B$ rather than between $A$ amd $B$. A similar story can be said with the rows of $A$.

]
, 
title: 
[
Warning
]
, 
background_color: 
rgb("#fcefdc")
, 
icon_color: 
rgb("#EB9113")
, 
icon: 
"⚠"
, 
body_background_color: 
white
)
]
=== Other `ScaledNormalization`'s
<other-scalednormalizations>
This trick can also apply to other normalization constraints. For example, we may want the maximum magnitude of each row of $B$ to one. This could make sense in applications where each row represents a waveform audio file (WAV) which has an audio format that takes values between $- 1$ and $1$. Instead of the Euclidean projection#footnote[In audio processing, this is commonly called "clipping".]

$ v arrow.l max (- 1 \, min (1 \, v)) $

onto the infinity ball

$ cal(B)_J (oo) = {v in bb(R)^J mid(bar.v) max_(j in [J]) abs(v [j]) lt.eq 1} \, $

we can apply the rescaling#footnote[In audio processing, this is commonly called "normalizing". Normalizing is often preferred to clipping since it maintains the perceived audio, but just at a different volume than the original signal. Clipping often introduces undesirable distortion (think of sound from a megaphone).]

$ v arrow.l frac(v, max_(j in [J]) abs(v [j])) . $

Applying this principle to the factorization problem

#math.equation(block: true, numbering: "(1)", [ $ min_(A in bb(R)^(I times R)\
B in bb(R)^(R times J)) 1 / 2 norm(A B - Y)_F^2 quad upright("s.t.") quad B in cal(B)_J^R (oo) = {B in bb(R)^(R times J) mid(bar.v) forall r in [R] \, thin max_(j in [J]) abs(B [r \, j]) lt.eq 1} $ ])<eq-B-row-infinity-problem>

gives us the update

#math.equation(block: true, numbering: "(1)", [ $ A & arrow.l A - 1 / L_A nabla_A f (A \, B)\
B & arrow.l B - 1 / L_B nabla_B f (A \, B)\
C [r \, r] & arrow.l max_(j in [J]) abs(B [r \, j]) quad upright("(") C in bb(R)^(R times R) upright(" is diagonal)")\
A & arrow.l A C\
B & arrow.l C^(- 1) B . $ ])<eq-nnpr-gd-B-rows-infinity>

This algorithm can be called in BlockTensorFactorization.jl with the following code.

```julia
options = (
    model=Tucker1,
    constraints=[linftyscale_rows! ∘ nonnegative!, noconstraint],
)

decomposition, stats, kwargs = factorize(Y; options...);
```

A similar story can be made about other $p$-norm balls $cal(B) (p)$ or spheres.

=== Other Tensor Factorizations
<other-tensor-factorizations>
The rescaling trick is applicable to other tensor factorizations, but is dependent on the exact model and constraints.

The simplest extension of matrix factorization is the Tucker-1 factorization of an order $N$ tensor $Y = B times_1 A$.

If we would like the first order slices of $B$ to be constrained to the simplex

$ B [r \, : \, :] in Delta_(J K) & = {B [r \, : \, :] in bb(R)_(+)^(J times K) mid(bar.v) sum_(j \, k) B [r \, j \, k] = 1}\
B in Delta_(J K)^R & = {B in bb(R)_(+)^(R times J times K) mid(bar.v) forall r in [R] \, thin sum_(j \, k) B [r \, j \, k] = 1} \, $

we can solve the problem

#math.equation(block: true, numbering: "(1)", [ $ min_(A in bb(R)^(I times R)\
B in bb(R)^(R times J times K)) 1 / 2 norm(B times_1 A - Y)_F^2 quad upright("s.t.") quad B in Delta_(J K)^R . $ ])<eq-B-slice-simplex-problem>

by iterating the update

#math.equation(block: true, numbering: "(1)", [ $ A & arrow.l A - 1 / L_A nabla_A f (A \, B)\
B & arrow.l B - 1 / L_B nabla_B f (A \, B)\
C [r \, r] & arrow.l sum_(j in [J] \, k in [K]) B [r \, j \, k] quad upright("(") C in bb(R)^(R times R) upright(" is diagonal)")\
A & arrow.l A C\
B & arrow.l B times_1 C^(- 1) . $ ])<eq-nnpr-gd-B-slices-simplex>

This algorithm can be called in BlockTensorFactorization.jl with the following code.

```julia
options = (
    model=Tucker1,
    constraints=[l1scale_1slices! ∘ nonnegative!, noconstraint],
)

decomposition, stats, kwargs = factorize(Y; options...);
```

A setting where this model applies would be if the first order slices of $B$ represent 2-dimensional probability densities.#footnote[If sampling a 2-dimensional probability density $p_r (x \, y)$ on a rectangular grid, entries of $B$ could be interpreted as $B [r \, j \, k] = p_r (x_j \, y_k) (x_j - x_(j - 1)) (y_k - y_(k - 1))$.]

For constraints that constrain an entire factor to some scale, the weight of that factor can be distributed to one or multiple other factors. For example, we may wish to find a CP-decomposition of an order 4-tensor $Y = lr(bracket.l.double A \, B \, C \, D bracket.r.double)$ where the Frobenius norm of $A$ is $1$. In the rescaling step, we could move the norm of $A$ to just the matrix $B$

$ c arrow.l norm(A)_F\
A arrow.l c^(- 1) A\
B arrow.l c B $

or all three other factors equally

$ c arrow.l norm(A)_F\
A arrow.l c^(- 1) A\
B arrow.l c^(1 \/ 3) B\
C arrow.l c^(1 \/ 3) C\
D arrow.l c^(1 \/ 3) D . $

In either case, the recombined tensor remains unchanged

$ lr(bracket.l.double c^(- 1) A \, c B \, C \, D bracket.r.double) = lr(bracket.l.double c^(- 1) A \, c^(1 \/ 3) B \, c^(1 \/ 3) C \, c^(1 \/ 3) D bracket.r.double) = lr(bracket.l.double A \, B \, C \, D bracket.r.double) . $

The two algorithms can be called in BlockTensorFactorization.jl with the following code.

```julia
options = (
    model=CPDecomposition,
    constraints=ConstraintUpdate(1, l2scaled!;
        whats_rescaled=(Y -> factor(Y, 2))), # B is the second factor
)

decomposition, stats, kwargs = factorize(Y; options...);
```

```julia
options = (
    model=CPDecomposition,
    constraints=ConstraintUpdate(1, l2scaled!),
    # assumes all other factors are rescaled
)

decomposition, stats, kwargs = factorize(Y; options...);
```

== Constraining Multiple Factors
<constraining-multiple-factors>
When constraining multiple factors with the rescaling approach, there must be at least one factor that is not constrained with rescaling.

Consider the following matrix factorization $Y = A B$ problem where we want both the columns of $A$ and rows of $B$ to be constrained to the simplex.#footnote[We define the constraint on $A$ in terms of $A^tack.b$ to be consistent with the simplex constraint defined on $B$.]

#math.equation(block: true, numbering: "(1)", [ $  & min_(A in bb(R)^(I times R)\
B in bb(R)^(R times J)) 1 / 2 norm(A B - Y)_F^2\
 & upright("s.t.")\
quad A^tack.b in Delta_I^R & = {A^tack.b in bb(R)_(+)^(R times I) mid(bar.v) forall r in [R] \, thin sum_i A^tack.b [r \, i] = 1}\
quad B in Delta_J^R & = {B in bb(R)_(+)^(R times J) mid(bar.v) forall r in [R] \, thin sum_j B [r \, j] = 1} $ ])<eq-AB-row-col-simplex-problem>

If we try to rescale both $A$ and $B$ while moving the weights to the other factor, we often observe numerical instability or blow up. To be clear, the following update does not seem to work in practice.

$ A & arrow.l max (0 \, A - 1 / L_A nabla_A f (A \, B))\
C_A [r \, r] & arrow.l sum_(i in [I]) A [i \, r]\
A & arrow.l A C_A^(- 1)\
B & arrow.l C_A B\
B & arrow.l max (0 \, B - 1 / L_B nabla_B f (A \, B))\
C_B [r \, r] & arrow.l sum_(j in [J]) B [r \, j]\
A & arrow.l A C_B\
B & arrow.l C_B^(- 1) B $

The relevant call in BlockTensorFactorization would be the following.

```julia
options = (
    model=Tucker1,
    constraints=[ # B is the 0th factor, A is the 1st factor
        ConstraintUpdate(1, l1scale_cols! ∘ nonnegative!;
            whats_rescaled=(x -> eachrow(factor(x, 0)))),
        ConstraintUpdate(0, l1scale_rows! ∘ nonnegative!;
            whats_rescaled=(x -> eachcol(factor(x, 1)))),
    ],
)

decomposition, stats, kwargs = factorize(Y; options...);
```

Instead, one of the factors can be scaled without moving the weight to the other factor. For example, if we wanted to remove the 4th line $B arrow.l C_A B$, we can call the following.

```julia
options = (
    model=Tucker1,
    constraints=[ # B is the 0th factor, A is the 1st factor
        ConstraintUpdate(1, l1scale_cols! ∘ nonnegative!;
            whats_rescaled=nothing),
        ConstraintUpdate(0, l1scale_rows! ∘ nonnegative!;
            whats_rescaled=(x -> eachcol(factor(x, 1)))),
    ],
)

decomposition, stats, kwargs = factorize(Y; options...);
```

This approach seems to work better in practice. The same principle of relaxing how constraints are enforced can be applied to the very similar problem

#math.equation(block: true, numbering: "(1)", [ $  & min_(A in bb(R)^(I times R)\
B in bb(R)^(R times J)) 1 / 2 norm(A B - Y)_F^2\
 & upright("s.t.")\
quad A in Delta_R^I & = {A in bb(R)_(+)^(I times R) mid(bar.v) forall i in [I] \, thin sum_r A [i \, r] = 1}\
quad B in Delta_J^R & = {B in bb(R)_(+)^(R times J) mid(bar.v) forall r in [R] \, thin sum_j B [r \, j] = 1} \, $ ])<eq-AB-row-simplex-problem>

where we want both the rows of $A$ and $B$ to be constrained to the simplex. Here, we cannot move the weights from $A$ to $B$ since there are $I$ rows of $A$ but only $R$ rows of $B$. Instead, we relax the problem to

#math.equation(block: true, numbering: "(1)", [ $  & min_(A in bb(R)^(I times R)\
B in bb(R)^(R times J)) 1 / 2 norm(A B - Y)_F^2\
 & upright("s.t.")\
 & quad A in bb(R)_(+)^(I times R)\
quad B in Delta_J^R & = {B in bb(R)_(+)^(R times J) mid(bar.v) forall r in [R] \, thin sum_j B [r \, j] = 1} . $ ])<eq-AB-row-simplex-problem-relaxed>

The relevant update for this relaxed problem would be

$ A & arrow.l max (0 \, A - 1 / L_A nabla_A f (A \, B))\
B & arrow.l max (0 \, B - 1 / L_B nabla_B f (A \, B))\
C_B [r \, r] & arrow.l sum_(j in [J]) B [r \, j]\
A & arrow.l A C_B\
B & arrow.l C_B^(- 1) B $

and is called in BlockTensorFactorization.jl with the following code.

```julia
options = (
    model=Tucker1,
    constraints=[ # B is the 0th factor, A is the 1st factor
        ConstraintUpdate(1, nonnegative!),
        ConstraintUpdate(0, l1scale_rows! ∘ nonnegative!;
            whats_rescaled=(x -> eachcol(factor(x, 1)))),
    ],
)

decomposition, stats, kwargs = factorize(Y; options...);
```

We justify this relaxation with the following argument. When the rows of $Y$ are in the simplex, we can bound how close the rows of $A$ are to summing to one with #ref(<thm-closeness-to-simplex>, supplement: [Theorem]).

#theorem("Closeness of A's rows summing to one")[
Let $Y in Delta_J^I$, $A in bb(R)_(+)^(I times R)$, and $B in Delta_J^R$ where

$ norm(Y - A B)_F lt.eq epsilon.alt . $

Then for any row $i in [I]$, the sum of the row $A [i \, :]$ is $epsilon.alt sqrt(J)$ close to 1

$ abs(1 - sum_(r in [R]) A [i \, r]) lt.eq epsilon.alt sqrt(J) . $

#block[
#emph[Proof]. We have the following inequalities.

$ norm(Y - A B)_F & lt.eq epsilon.alt\
1 / sqrt(J) norm(Y - A B)_oo & lt.eq epsilon.alt quad (1 / sqrt(J) norm(X)_oo lt.eq norm(X)_2 lt.eq norm(X)_F med upright("for") med X in bb(R)^(I times J))\
1 / sqrt(J) max_(i in [I]) (sum_(j in [J]) abs((Y - A B) [i \, j])) & lt.eq epsilon.alt\
max_(i in [I]) (sum_(j in [J]) abs((Y - A B) [i \, j])) & lt.eq epsilon.alt sqrt(J) $ So for all rows $i in [I]$,

$ epsilon.alt sqrt(J) & gt.eq sum_(j in [J]) abs((Y - A B) [i \, j])\
 & gt.eq abs(sum_(j in [J]) (Y - A B) [i \, j])\
 & = abs(sum_(j in [J]) Y [i \, j] - sum_(j in [J]) (A B) [i \, j])\
 & = abs(1 - sum_(j in [J]) sum_(r in [R]) A [i \, r] B [r \, j]) quad (upright("since") med Y in Delta_J^I)\
 & = abs(1 - sum_(r in [R]) A [i \, r] (sum_(j in [J]) B [r \, j]))\
 & = abs(1 - sum_(r in [R]) A [i \, r]) quad (upright("since") med B in Delta_J^R) . $

]
] <thm-closeness-to-simplex>
#ref(<thm-closeness-to-simplex>, supplement: [Theorem]) implies that solutions to the relaxed problem (#ref(<eq-AB-row-simplex-problem-relaxed>, supplement: [Equation])) are approximate solutions to the problem shown in #ref(<eq-AB-row-simplex-problem>, supplement: [Equation]). Moreover, if there exists an exact factorization $Y = A B$ for the relaxed problem, then it is also a solution the original problem.

== Experiment
<experiment>
To illustrate the advantage of this rescaling trick over Euclidean projection, we will consider solving the problem shown in #ref(<eq-AB-row-simplex-problem>, supplement: [Equation]) where $Y in Delta_J^I$ and there exists an exact factorization $Y = A B$ with $A in Delta_R^I$ and $B in Delta_J^R$ using eight different algorithms.

= Multi-scale
<sec-multi-scale>
- use a coarse discretization along continuous dimensions
- factorize
- linearly interpolate decomposition to warm start larger decompositions

In many applications @saylor_CharacterizingSedimentSources_2019 \[TODO add spatial transcriptomics\], the data tensor $Y$ represents a discretization of continuous data. In these cases, we can decide how finely or coarsely to sample the data and effectively have control over the dimension of tensor needing to be factored.

For high quality results, we would like as fine of a discretization as possible. Or if the data is already collected in a discretized form, we would like to incorporate all the data. But smaller tensors are faster to decompose because there are fewer parameters to fit and learn. Most matrix operations like addition, multiplication, and finding their norm are also faster for smaller tensors.

We propose a multiresolution approach inspired wavelets @benedetto_wavelets_1993 and multigrid @trottenberg_multigrid_2001 that factorizes the data at progressively finer scales. This can greatly speed up how fast large scale tensors can be factorized.

== Basic Approach
<sec-basic-multi-scale>
We will start with same example from the beginning of #ref(<sec-ppr>, supplement: [Section]). Give a tensor $Y in bb(R)_(+)^(I times J)$ representing mixtures of discretized densities, we would like to demix the densities according to the model

$ mat(delim: "[", arrow.l, med y_1^tack.b, arrow.r; arrow.l, med y_2^tack.b, arrow.r; arrow.l, med dots.v, arrow.r; arrow.l, med y_I^tack.b, arrow.r) & = mat(delim: "[", a_11, a_12, dots.h, a_(1 R); a_21, a_22, dots.h, a_(2 R); dots.v, , , dots.v; a_(I 1), a_(I 2), dots.h, a_(I R); #none) mat(delim: "[", arrow.l, med b_1^tack.b, arrow.r; arrow.l, med b_2^tack.b, arrow.r; med, dots.v, med; arrow.l, med b_R^tack.b, arrow.r) $

or $ Y = A B . $

Notice that the rows of $Y$ and $B$ represent samples of continuous densities, so the size of their second dimension $J$ is arbitrary. Suppose each of the 1-dimensional densities are uniformly discretized on an interval $[a \, b]$ with $J_s$ number of points. We use $s$ to represent the scale or spacing of the number of points. For example, $J_1 = J$ would be the finest scale using every point in a discretization $x_1 \, x_2 \, x_3 dots.h \, x_(J_1)$ with $x_1 = a$ and $x_(J_1) = b$, $J_2 = J_1 \/ 2$ would be coarser and use every other point $x_1 \, x_3 \, x_5 \, dots.h x_(J_1 - 1)$.#footnote[We assume $J$ is even here, but we could define $J_2 = (J_1 - 1) \/ 2$ if $J_1 = J$ is odd.]

The basic approach is to factorize $Y_2 in bb(R)^(I times J_2)$ with entries $Y_2 [i \, j] = y_i (x_(2 j - 1))$ to obtain $A_2^(T_2) in bb(R)^(I times R)$ and $B_(""^(T_2)) in bb(R)^(I times J_2)$ after $T_2$ many iterations. We use the factors $A_2^(T_2)$ and $B_2^(T_2)$ to initialize the factorization of $Y_1 = Y in bb(R)^(I times J_1)$. We can initialize $A_1^0 = A_2^(T_2)$ since the size of $A$ is the same at both scales, and repeat every entry of $B$ to initialize $B_1^0$ with entries $B_1^0 [i \, j] = B_2^(T_2) [i \, ceil.l j \/ 2 ceil.r]$.

Factorizing $Y_2$ is faster than factorizing $Y_1$ since 1) there are fewer parameters to learn#footnote[Only $(I + J_2) R$ at the coarse scale which is less than $(I + J_1) R$ for the fine scale.] and 2) most arithmetic like addition and multiplication, as well as calling other operators like `norm` are faster to compute. This gives us a better initialization for $A$ and $B$ in the factorization of $Y_1$ than some other random initialization so that fewer iterations are needed at the more expensive finer scale.

== General Approach
<general-approach>
The basic approach can be generalized in two ways: the data could be continuous in multiple dimensions, and we can recursively apply this multi-scale approach to progressively refine a very coarse factorization.

Suppose we are given a tensor $Y in bb(R)^(I_1 times dots.h.c times I_N)$ where the dimensions $I_(n_1) \, dots.h \, I_(n_M)$ represent a grided discretization of $M$-dimensional continuous data.

An example of this setting would be an extension of the example shown in #ref(<sec-basic-multi-scale>, supplement: [Section]) to higher dimensional distributions. We could consider an order-$3$ tensor where the horizontal slices correspond to a $2$-dimensional discretization of a bivariate density. Entries of the input tensor $Y$ would be given by $Y [i \, j \, k] = f_i (x_j \, y_k)$ for continuous probability density functions $f_i : bb(R)^2 arrow.r bb(R)_(+)$ for a 2D grid of points $(x_j \, y_k)$. In this example, the second and third dimensions would be continuous.

If we want to perform a rank-$(R_1 \, dots.h \, R_N)$ Tucker decomposition of $Y = ⟦A_0 \; A_1 \, dots.h \, A_N⟧$, we can initialize the factors with a very corse factorization $A_(n_m)^s in bb(R)^(J_(n_m)^s times R_(n_m))$ where $J^s$ would be a discretization that uses every $2^s$ points, or more accurately

$ J_(n_m)^s = 2^(max (S_m - s \, 0)) + 1 $

points in total. We select $S_m$ so that at the finest scale $s = 0$, we have $J_(n_m)^0 = I_(n_m)$. The dimensions that do not represent continuous values will use the full sized factors $A_n^s in bb(R)^(J_n times R_n)$ with $J_n = I_n$, and the core will remain $A_0^s in bb(R)^(R_1 times dots.h.c times R_N)$ at all scales $s$.#footnote[This assumes the dimensions of $Y$ where $Y$ is continuous have been discretized as one more than a power of two. The same idea holds with some other discretization plan, but becomes more complicated to express notationally and keep track of the number of points at each scale.]

The aim is to fit the product of the factors $⟦A_0^s \; A_1^s \, dots.h \, A_N^s⟧$ to a lower resolution version of $Y in bb(R)^(I_1 times dots.h.c times I_N)$, namely $Y^s in bb(R)^(J_1^s times dots.h.c times J_N^s)$, and use the result to initialize a finer version of the factors $A$. The code for this looks like the following.

```julia
function multiscale_factorize(Y; kwargs...)
    continuous_dims, kwargs = initialize_continuous_dims(Y; kwargs...)
    scales, kwargs = initialize_scales(Y; kwargs...)
    coarsest_scale, finer_scales... = scales

    # Factorize Y at the coarsest scale
    Yₛ = coarsen(Y, coarsest_scale; dims=continuous_dims, kwargs...)

    constraints, kwargs = scale_constraints(Yₛ, coarsest_scale; kwargs...)
    decomposition, stats, _ = factorize(Yₛ; kwargs...)

    # Factorize Y at progressively finer scales
    for scale in finer_scales
        # Use an interpolated version of the coarse factorization
        # as the initialization.
        decomposition = interpolate(decomposition, 2; dims=continuous_dims, kwargs...)
        kwargs[:decomposition] = decomposition

        Yₛ = coarsen(Y, scale; dims=continuous_dims, kwargs...)

        constraints, kwargs = scale_constraints(Yₛ, scale; kwargs...)

        decomposition, stats, _ = factorize(Yₛ; kwargs...)
    end
    return decomposition, stats, kwargs
end
```

=== Coarsening and Interpolating
<coarsening-and-interpolating>
Straightforward subsampled coarsening and constant interpolating can be used for `coarsen` and `interpolate`, but more sophisticated methods can be used in principle. Since the final solve of `factorize` is on the original sized problem, the choice of coarsening and interpolating only influences the initialization used at this finest scale. Bellow are examples of the basic coarsening and interpolation methods.

```julia
coarsen(Y::AbstractArray, scale::Integer; dims=1:ndims(Y), kwargs...) =
    Y[(d in dims ? axis[begin:scale:end] : axis for (d, axis) in enumerate(axes(Y)))...]

function interpolate(Y, scale; dims=1:ndims(Y), kwargs...)
    Y = repeat(Y; inner=(d in dims ? scale : 1 for d in 1:ndims(Y)))

    # Chop the last slice of repeated dimensions
    # since we only interpolate between the values
    return Y[(d in dims ? axis[begin:end-scale+1] : axis for (d, axis) in enumerate(axes(Y)))...]
end
```

To apply a linear interpolation, we can first perform the constant interpolation and smooth out the result. The following code averages neighbouring values along the continuous dimensions specified. Since this is applied after the constant interpolation, every even indexed value becomes the average of its neighbours, and every odd indexed value remains fixed.

```julia
function linear_smooth!(Y, dims)
    all_dims = 1:ndims(Y)
    for d in dims
        axis = axes(Y, d)
        Y1 = @view Y[(i==d ? axis[begin+1:end-1] : (:) for i in all_dims)...]
        Y2 = @view Y[(i==d ? axis[begin+2:end] : (:) for i in all_dims)...]

        @. Y1 = 0.5 * (Y1 + Y2)
    end
    return Y
end
```

When interpolating an array that is an `AbstractDecomposition`, we can interpolate the factors directly instead of the combined array.

```julia
function interpolate(CPD::CPDecomposition, scale; dims=1:ndims(CPD), kwargs...)
    interpolated_matrix_factors = (d in dims ? interpolate(A, scale; dims=1, kwargs...) : A for (d, A) in enumerate(matrix_factors(CPD)))
    return CPDecomposition(Tuple(interpolated_matrix_factors))
end

function interpolate(T::Tucker1, scale; dims=1:ndims(T), kwargs...)
    core_dims = setdiff(dims, 1) # Want all dimensions except possibly the first
    interpolated_core = interpolate(core(T), scale; dims=core_dims, kwargs...)

    matrix = matrix_factor(T, 1)

    interpolated_matrix = 1 in dims ? interpolate(matrix, scale; dims=1, kwargs) : matrix
    return Tucker1((interpolated_core, interpolated_matrix))
end

function interpolate(T::Tucker, scale; dims=1:ndims(T), kwargs...)
    interpolated_matrix_factors = (d in dims ? interpolate(A, scale; dims=1, kwargs...) : A for (d, A) in enumerate(matrix_factors(T)))
    # Core is not interpolated
    return Tucker(Tuple(core(T), interpolated_matrix_factors...))
end
```

== Constraints with Multi-scale
<constraints-with-multi-scale>
Some constraints like `Entrywise` constrains can be used as-is at any scale, but other constraints like normalizations and linear constraints require some more thought. The main strategy is to modify constraints along continuous dimensions. Given a constraint on a factor, and the number of continuous dimensions it constrains, we can construct a modified constraint by scaling the full sized constraint appropriately.

For a $p$-norm constraint where some part of a factor $X_i$ needs to be normalized to $lr(bar.v.double X_i bar.v.double)_p = C$, we require the coarsened parts of the factor $overline(X)_i$ to have norm $lr(bar.v.double X_i bar.v.double)_p = (C \/ s)^(1 \/ p)$ where $s$ is the scale of the coarsening. For example, a scale of $3$ would mean we only take every $3$ entries of $X_i$.

For linear constraints `AX=B`, we construct a new constraint that coarsens $A$, and scales the bias $B$ by the scale. For example, if $x$ is a vector that represents a continuous function that is constrained to the affine space $A x = b$, we treat the rows of the matrix $A$ as also being continuous functions that are inner product-ed with $x$ and can be coarsened in a similar manner to $x$. This means a `LinearConstraint(A, b)` gets scaled to the constraint `LinearConstraint(A[:, begin:scale:end], b ./ scale)`.

In all the constraints, the scale will need to be applied in each continuous dimension, so we implement this as `scale^n_continuous_dims`. The full implementation of `scale_constraint` is shown below.

```julia
function scale_constraint(constraint::AbstractConstraint, scale, n_continuous_dims)
    @warn "Unsure how to scale constraints of type $(typeof(constraint)). Leaving the constraint $constraint alone."
    return constraint
end

# Constraints that are fixed like entrywise constraints do not need to be scaled
function scale_constraint(constraint::Union{FIXED_CONSTRAINTS...}, scale, n_continuous_dims)
    return constraint
end

# TODO handle LinearConstraint{<:AbstractArray} or LinearConstraint{Function}
function scale_constraint(constraint::LinearConstraint{<:AbstractMatrix}, scale, n_continuous_dims)
    A = constraint.linear_operator
    b = constraint.bias
    return LinearConstraint(A[:, begin:scale:end], b ./ scale^n_continuous_dims)
end

function scale_constraint(constraint::ScaledNormalization{<:Union{Real,AbstractArray{<:Real}}}, scale, n_continuous_dims)
    norm = constraint.norm
    F = constraint.whats_normalized
    S = constraint.scale
    return ScaledNormalization(norm, F, S ./ scale^n_continuous_dims)
end

function scale_constraint(constraint::ScaledNormalization{<:Function}, scale, n_continuous_dims)
    norm = constraint.norm
    F = constraint.whats_normalized
    S = constraint.scale
    return ScaledNormalization(norm, F, (x -> x ./ scale^n_continuous_dims) ∘ S)
end

# TODO scale a projected normalization
function scale_constraint(constraint::ProjectedNormalization, scale, n_continuous_dims)
    @warn "Scaling ProjectedNormalization constraints is not implemented (YET!) Leaving the constraint $constraint alone."
    return constraint
end
```

We justify scaling constraints in this way with the following propositions and observations.

#theorem("Constraint Proposition Assumptions")[
In the following proposition, we assume $x in bb(R)^I$ represents an evenly-spaced discretization of an $L_f$-Lipschitz function $f : [l \, u] arrow.r bb(R)$ on a finite interval $t in [l \, u]$ with entries

$ x [i] = f (t [i]) \, $

where

$ t [i] = l + (i - 1) Delta t = l + (i - 1) frac(u - l, I - 1) . $

We will use $overline(x) in bb(R)^(floor.l (I + 1) \/ 2 floor.r)$ to represent the subvector made by removing every other entry of $x in bb(R)^I$ where

$ overline(x) = (x [1] \, x [3] \, x [5] \, dots.h \, x [floor.l I floor.r_o]) . $

We use the shorthand

$ floor.l I floor.r_o = 2 ⌊frac(I + 1, 2)⌋ - 1 $

to round $I$ down to the nearest odd integer.

] <thm-rem-constraint-assumptions>
#proposition("Linear Constraint Scaling")[
Let the assumption in #ref(<thm-rem-constraint-assumptions>, supplement: [Theorem]) hold. Let $a in bb(R)^I$ be a discretization of an $L_g$-Lipschitz function $g : [l \, u] arrow.r bb(R)$.

If $lr(angle.l a \, x angle.r) = b$, then for even $I$,

$ abs(⟨overline(a) \, overline(x)⟩ - b \/ 2) lt.eq max (lr(bar.v.double f bar.v.double)_oo \, lr(bar.v.double g bar.v.double)_oo) I / (I - 1) frac((L_f + L_g) (u - l), 4) $

and for odd $I$,

TODO check formatting $ abs(2 norm(overline(x))_1 - frac(I + 1, I) lr(bar.v.double x bar.v.double)_1) & = abs(2 norm(overline(x))_1 - lr(bar.v.double x bar.v.double)_1 - lr(bar.v.double x bar.v.double)_1 / I)\
 & = abs(2 sum_(i med upright("odd")) lr(|x_i) - sum_(i = 1)^I abs(x_i) - lr(bar.v.double x bar.v.double)_1 / I|)\
 & = abs(sum_(i med upright("odd")) lr(|x_i) + sum_(i med upright("odd")) abs(x_i) - (sum_(i med upright("odd")) abs(x_i) + sum_(i med upright("even")) abs(x_i)) - lr(bar.v.double x bar.v.double)_1 / I|)\
 & = abs(sum_(i med upright("odd")) lr(|x_i) - sum_(i med upright("even")) abs(x_i) - lr(bar.v.double x bar.v.double)_1 / I|)\
 & = abs(sum_(j = 1)^((I - 1) \/ 2) lr(|x_(2 j - 1)) + abs(x_I) - sum_(j = 1)^((I - 1) \/ 2) abs(x_(2 j)) - lr(bar.v.double x bar.v.double)_1 / I|)\
 & lt.eq sum_(j = 1)^((I - 1) \/ 2) abs(lr(|x_(2 j - 1)) - abs(x_(2 j))|) + abs(lr(|x_I) - lr(bar.v.double x bar.v.double)_1 / I|)\
 & lt.eq sum_(j = 1)^((I - 1) \/ 2) abs(x_(2 j - 1) - x_(2 j)) + 1 / I abs(I lr(|x_I) - lr(bar.v.double x bar.v.double)_1|)\
 & lt.eq sum_(j = 1)^((I - 1) \/ 2) C + 1 / I abs(sum_(i = 1)^I (lr(|x_I) - abs(x_i))|)\
 & lt.eq C frac(I - 1, 2) + 1 / I sum_(i = 1)^I abs(lr(|x_I) - abs(x_i)|)\
 & lt.eq C frac(I - 1, 2) + 1 / I sum_(i = 1)^I abs(x_I - x_i)\
 & lt.eq C frac(I - 1, 2) + 1 / I sum_(i = 1)^I C (I - i) med upright("(Apply Lipschitz recursively)")\
 & = C frac(I - 1, 2) + 1 / I frac(C I (I - 1), 2)\
 & = frac(L (u - l), 2) + frac(L (u - l), 2)\
 & = L (u - l) . $

#block[
#emph[Proof]. See #ref(<sec-linear-constraint-scaling-proof>, supplement: [Section]).

]
] <prp-linear-constraint-scaling>
#ref(<prp-linear-constraint-scaling>, supplement: [Proposition]) can be extended to a bound on the $L_1$ distance between $overline(A) overline(x)$ and $b \/ 2$ where $A x = b$ and $overline(A) = A [: \, b e g i n : 2 : e n d]$ is the submatrix with every other column removed. TODO add corollary.

We also have a similar proposition for a $1$-norm constraint.

#proposition("$L_1$-norm Constraint Scaling")[
Let the assumption in #ref(<thm-rem-constraint-assumptions>, supplement: [Theorem]) hold.

If $lr(bar.v.double x bar.v.double)_1 = b$, then for even $I$,

$ abs(norm(overline(x))_1 - b \/ 2) lt.eq I / (I - 1) frac(L (u - l), 4) $

and for odd $I$,

$ abs(norm(overline(x))_1 - frac(I + 1, I) b / 2) lt.eq frac(L (u - l), 2) . $

#block[
#emph[Proof]. See #ref(<sec-l1-norm-constraint-scaling-proof>, supplement: [Section]).

]
] <prp-l1-norm-constraint-scaling>
Importantly, for normalized vectors $z = x \/ lr(bar.v.double x bar.v.double)_1$, the error bound in #ref(<prp-l1-norm-constraint-scaling>, supplement: [Proposition]) (for even $I$) becomes

$ abs(norm(overline(z))_1 - 1 \/ 2) lt.eq I / (I - 1) frac(L (u - l), 4 b) . $

As the number of points get large, $I arrow.r oo$, a finer discretization will have more points and have a $1$-norm becoming unbounded $b arrow.r oo$. This means the error bound goes to zero and $norm(overline(z))_1 arrow.r 1 \/ 2$.#footnote[This also holds for odd $I$.]

== Convergence of a Multi-scale Method
<convergence-of-a-multi-scale-method>
- show the multi-scale method has tighter bounds than regular gradient descent for lipschitz data
- this assumes no constraints

As implemented, both the regular and multi-scale approaches execute many iterations at the finest scale. The multi-scale method uses its coarsened iterations to warm start the fine iteration. So we should expect the multi-scale method to converge. We make this explicit with the following lemmas and theorems. The first subsection (#ref(<sec-analysis-lemmas>, supplement: [Section])) shows general functional and convex analysis facts about Lipschitz, smooth, and strongly convex functions, and what happens when you linearly interpolate Lipschitz functions. The next subsection (#ref(<sec-resolved-multiscale>, supplement: [Section])) shows convergence for the multi-scale method when we resolve the problem along the entire new discretization at each scale. And the final subsection (#ref(<sec-freezed-multiscale>, supplement: [Section])) looks at what happens when you freeze previously computed points so you only need to solve the problem at the interpolated points for each scale.

=== Definitions
<definitions>
#definition()[
A differentiable function $f : bb(R)^I arrow.r bb(R)$ is $S$-smooth when $nabla f$ is $S$-Lipschitz,

$ lr(bar.v.double nabla f (x) - nabla f (y) bar.v.double)_2 lt.eq S lr(bar.v.double x - y bar.v.double)_2 . $

] <def-smooth-function>
#definition()[
A function $f : bb(R)^I arrow.r bb(R)$ is $mu$-strongly convex when $g (x) = f (x) - mu / 2 lr(bar.v.double x bar.v.double)_2^2$ is convex.

When $f$ is differentiable, $f$ is $mu$-strongly convex when

$ f (y) gt.eq f (x) + ⟨nabla f (x) \, y - x⟩ + mu / 2 norm(x - y)_2^2 $

for all $x \, y in bb(R)^I$.

] <def-strongly-convex-function>
#definition()[
One step of projected gradient descent for an $S$-smooth and $mu$-strongly convex function $cal(L) : bb(R)^I arrow.r bb(R)$, a constraint set $cal(C) subset.eq V$, at an iterate $x^k$ is the update,

$ x^(k + 1) arrow.l P_(cal(C)) (x^k - 1 / S nabla cal(L) (x^k)) $

where $P_(cal(C)) : bb(R)^I arrow.r cal(C)$ is the (Euclidean) projection operator onto $cal(C)$.

] <def-projected-gradient-decent>
#proposition("Descent Lemma")[
In the projected gradient descent setting with an $S$-smooth and $mu$-strongly convex function (#ref(<def-projected-gradient-decent>, supplement: [Definition])), we can bound the distance to the unique minimizer $x^(\*) in cal(C)$ in terms of the initial point $x^0 in bb(R)^I$,

$ norm(x^t - x^(\*))_2 lt.eq (1 - c)^t norm(x^0 - x^(\*))_2 $

with #emph[condition number] $c = mu \/ S$.

] <prp-descent-lemma>
TODO cite this.

The vector space $bb(R)^I$ in #ref(<def-strongly-convex-function>, supplement: [Definition]), #ref(<def-smooth-function>, supplement: [Definition]), and #ref(<def-projected-gradient-decent>, supplement: [Definition]) can be extended to tensor spaces $bb(R)^(I_1 times dots.h.c times I_N)$ with the Frobenius inner product $lr(angle.l X \, Y angle.r)_F$ and norm $lr(bar.v.double dot.op bar.v.double)_F$ more generally.

=== Functional Analysis Lemmas
<sec-analysis-lemmas>
Before we can analyze the convergence of a multi-scaled method, we first need a handle on how linear interpolations play with discretized Lipschitz functions. To simplify the analysis of multi-scaled methods, we look at solving the general problem

$ min_(x in cal(C)) cal(L) (x) := sum_(i in [I]) ell_i (x [i]) $

using projected gradient descent

$ x arrow.l P_(cal(C)) (x - 1 / L nabla cal(L) (x)) $

where the samples $x [i]$ come from some $L_f$-Lipschitz function $f : bb(R) arrow.r bb(R)$

$ x [i] = f (t_i) $

on an interval $t in [a \, b]$ that has been evenly discretized $t_(i + 1) - t_i = Delta t$ for all $i in [I]$.

We assume the loss functions $ell_i$ over each entry $x [i]$ are $S$-smooth and $mu$-strongly convex. A natural example of such a function $cal(L)$ would be a least-squares loss

$ cal(L) (x) = 1 / 2 norm(x - y)_2^2 = sum_(i in [I]) 1 / 2 (x [i] - y [i])^2 = sum_(i in [I]) ell_i (x [i]) . $

The following lemmas explain that the smoothness and strong convexity of the functions $ell_i$ carry over to the full loss $cal(L)$.

#lemma()[
If $ell_i : bb(R) arrow.r bb(R)$ are each $S$-smooth, then $cal(L) : bb(R)^I arrow.r bb(R)$ is $S$-smooth.

#block[
#emph[Proof]. First note that $(nabla cal(L) (x))_i = frac(partial, partial x [i]) cal(L) (x) = frac(partial, partial x [i]) sum_(j = 1)^I ell_i (x [j]) = sum_(j = 1)^I frac(partial, partial x [i]) ell_i (x [j]) = frac(partial, partial x [i]) ell_i (x [i]) = ell_i^(') (x [i])$ since $cal(L)$ is separable in each coordinate. This gives us, $ norm(nabla cal(L) (x) - nabla cal(L) (y))_2^2 & = sum_(i = 1)^I (nabla cal(L) (x) - nabla cal(L) (y)) [i]^2\
 & = sum_(i = 1)^I (ell_i^(') (x [i]) - ell_i^(') (x [i]))^2\
 & lt.eq sum_(i = 1)^I S^2 (x [i] - y [i])^2\
 & = S^2 norm(x - y)_2^2 . $

Taking square roots completes the proof.

]
] <lem-smooth-component-functions>
#lemma()[
If $ell_i : bb(R) arrow.r bb(R)$ are all $m$ strongly convex, then $cal(L) : bb(R)^I arrow.r bb(R)$ is $m$ strongly convex (under the $2$-norm in $bb(R)^I$).

#block[
#emph[Proof]. Consider $ cal(L) (x) - m / 2 lr(bar.v.double x bar.v.double)_2^2 & = sum_(i = 1)^I ell_i (x [i]) - m / 2 sum_(i = 1)^I x [i]^2\
 & = sum_(i = 1)^I (ell_i (x [i]) - m / 2 x [i]^2) . $

The function $g (x [i]) = ell_i (x [i]) - m / 2 x [i]^2$ is convex because each function $ell_i (x [i])$ is $m$ strongly convex. The sum of convex functions is also convex, so $cal(L) (x) - m / 2 lr(bar.v.double x bar.v.double)_2^2$ is convex.

]
] <lem-strongly-convex-component-functions>
These conditions can be relaxed to include larger sets of function like quasi-strongly convex functions @necoara_linear_2019, but the general proof approach remains the same. We do require the separability of $cal(L)$ so it can make sense to solve the problem over coarser discretizations.

Now we discuss how Lipschitz functions play with discretizations and linear interpolations.

#lemma("Lipschitz Function Interpolation")[
Let $f : bb(R)^n arrow.r bb(R)$, $a \, b in bb(R)^n$, $t in [0 \, 1]$. Suppose $f$ is $L$-Lipshtiz. Then the error between the function at a point on the line segment between $a$ and $b$, and the linear interpolation is $ abs((t f (a) + (1 - t) f (b)) - f (t a + (1 - t) b)) lt.eq 2 L t (1 - t) norm(a - b)_2 . $

] <lem-lipschitz-interpolation>
Using the bound on the linear interpolation of a Lipschitz function repeatedly, for centre point interpolation ($t = 1 \/ 2$ in #ref(<lem-lipschitz-interpolation>, supplement: [Lemma])), we can bound the error between an exact discretization of a function at a fine scale ($Y [j] = f (X [j])$) and a linear interpolation ($hat(Y)$) coming from a coarser discretization ($y [k] = f (x [k])$).

#lemma("Exact Interpolation")[
Given $K$ uniformly space points $x [k]$ on $[a \, b]$, (nearly) double them to get $J = 2 K - 1$ uniformly spaced points $X [j]$ on $[a \, b]$. Let $y [k] = f (x [k])$ for some $L$-Lipshitz function $f : bb(R) arrow.r bb(R)$, and linearly interpolate the function values

$ hat(Y) [j] = cases(delim: "{", y [frac(j + 1, 2)] & upright("if ") j upright(" is odd"), 1 / 2 (y [j / 2] + y [j / 2 + 1]) & upright("if ") j upright(" is even")) med \, $

where the true values are given by $Y [j] = f (X [j])$. Then the difference between the interpolated $hat(Y)$ and exact values $Y$ is bounded by $ norm(hat(Y) - Y)_2 lt.eq frac(L, 2 sqrt(K - 1)) norm(a - b)_2 . $

] <lem-exact-interpolation>
We also have an inexact version when we interpolate not from an exact coarse discretization ($y [k] = f (x [k])$), but from an approximate coarse discretization ($tilde(y) [k] = f (x [k]) + delta_k$).

#lemma("Inexact Interpolation")[
Given a linear interpolation $hat(tilde(Y))$ of an inexact discretization $tilde(y) [k] = f (x [k]) + delta_k$ of a function $f$ as described in #ref(<lem-exact-interpolation>, supplement: [Lemma]), where the interpolation is defined as

$ hat(tilde(Y)) [j] & = cases(delim: "{", tilde(y) [frac(j + 1, 2)] & upright("if ") j upright(" is odd"), 1 / 2 (tilde(y) [j / 2] + tilde(y) [j / 2 + 1]) & upright("if ") j upright(" is even")) \, $

we have the error bound between the interpolated inexact discretization $hat(tilde(Y))$ and the exact discretization of the function $Y$,

$ norm(hat(tilde(Y)) - Y)_2 & lt.eq norm(hat(tilde(Y)) - hat(Y))_2 + norm(hat(Y) - Y)_2\
 & lt.eq sqrt(2) norm(tilde(y) - y) + frac(L, 2 sqrt(K - 1)) norm(a - b)_2 . $

#block[
#emph[Proof]. See #ref(<sec-inexact-interpolation-proof>, supplement: [Section]).

]
] <lem-inexact-interpolation>
This makes sense in the following way: the error in our interpolation of approximate points, is bounded by two errors. The first comes from the fact that we interpolated using approximated values $tilde(y)$ in place of the true values $y$, and the second comes from using a linear interpolation $hat(Y)$ in place of the exact values $Y$.

=== Re-solved Multi-scale
<sec-resolved-multiscale>
LEFT OFF HERE

We use the following general approach to show convergence.

Given some initial guess $x_S^0$ at the coarsest scale, we use the decent lemma (#ref(<prp-descent-lemma>, supplement: [Proposition])) to bound the error between our iterate $x_S^(K_S)$ after $K_S$ iterations, and the solution $x_S^(\*)$ at the scale $S$. We can linearly interpolate our point $hat(x)_S^(K_S)$ and use it to initialize another round of projected gradient descent $x_(S - 1)^0 = hat(x)_S^(K_S)$ at the slightly finer scale. We can link the error between our iterate $x_S^(K_S)$ and the solution $x_S^(\*)$ at scale $S$ with the initial error between an iterate at the finer scale $x_(S - 1)^0$ and the solution $x_(S - 1)^(\*)$ at this scale using the inexact interpolation lemma (#ref(<lem-inexact-interpolation>, supplement: [Lemma])). We perform $K_s$ iterations at each scale $s = S \, S - 1 \, dots.h \, 2 \, 1$ to get an error bound between our iterate at the finest scale $x_1^K ""_1$ the solution at this scale $x_1^(\*)$ in terms of the initial error at the coarsest scale $x_S^0$.

We present the main descent theorem here, and leave the rest of the details in the appendix.

#theorem("Re-solved Multi-scale Descent Error")[
Let $c$ be the condition number of the loss function $cal(L)$, $L_f$ be the Lipschitz constant for the underlying continuous function $f$, and $S$ be the coarsest scale. For the setting described in #ref(<sec-analysis-lemmas>, supplement: [Section]), we have the following error bound on the fixed multi-scaled method. $  & norm(x_1^0 - x_1^(\*))_2\
 & lt.eq sqrt(2^(S - 1)) (1 - c)^(sum_(s = 1)^S K_s) norm(x_S^0 - x_S^(\*))_2 + frac(L_f abs(a - b), 2 sqrt(2^(S + 1))) sum_(s = 1)^(S - 1) 2^s (1 - c)^(sum_(t = 1)^s K_t) $

#block[
#emph[Proof]. See #ref(<sec-resolved-multiscale-descent-error-proof>, supplement: [Section])

]
] <thm-resolved-multiscale-descent-error>
=== Freezed Multi-scale
<sec-freezed-multiscale>
We take a nearly identical approach to the re-solved multi-scale method as described in #ref(<sec-resolved-multiscale>, supplement: [Section]), but we instead freeze the entries of our interpolation $hat(x)_s^(K_s)$ that correspond to the same points as the prior iteration $x_s^(K_s)$. The projected gradient update will only act on the interpolated and unfrozen values which we will call $x_((s))^k$.

TODO can I just introduce this notation in the proofs so that the main paper is cleaner?

To be clear let us look at an example where the finest scale has $2^3 + 1 = 9$ points. After performing $K_3$ iterations at the coarsest scale $S = 3$ with a total of three points, and slightly finer scale $S = 2$ with five points (but only two unfrozen points), the full vector at the finest scale would be

$  & x_1^k =\
 & (x_((3))^(K_3) [1] \, x_((1))^k [1] \, x_((2))^(K_2) [1] \, x_((1))^k [2] \, x_((3))^(K_3) [2] \, x_((1))^k [3] \, x_((2))^(K_2) [2] \, x_((1))^k [4] \, x_((3))^(K_3) [3]) $

where the vector of free variables is

$  & x_((1))^k = (x_((1))^k [1] \, x_((1))^k [2] \, x_((1))^k [3] \, x_((1))^k [4]) . $

#strong[Summary of notation]

- $hat(x)$ is some approximation of $x$
- $x_((s))$ is a vector of just the free variables at scale $s$
- $x_s$ is a vector of the free and fixed variables
- $x^k$ is the $k$th iteration
- $K_s$ is the number of iterations performed at scale $s$
- $e = hat(x) - x^(\*)$ is the error
- have $e_((s)) \, e_s \, e^k$ similarly.
  - Example, $e_1^2 = x_1^2 - x_1^(\*)$ is the error between our iterates at the finest scale $s = 1$ and the true values $x_1^(\*)$ after $2$ iterations

We present the main descent theorem here, and leave the rest of the details in the appendix.

#theorem("Freezed Multi-scale Descent Error")[
Let $c$ be the condition number of the loss function $cal(L)$. For the setting described in #ref(<sec-analysis-lemmas>, supplement: [Section]), we have the following error bound on the freezed multi-scaled method.

$ norm(e_1^(K_1)) & lt.eq norm(e_S^0) (1 - c)^(K_S) product_(s = 1)^(S - 1) (1 + (1 - c)^(K_s))\
 & #h(2em) + med L_f / 2 abs(b - a) sum_(s = 1)^(S - 1) 1 / sqrt(2^(S - s)) (1 - c)^(K_s) product_(j = 1)^(s - 1) (1 + (1 - c)^(K_j)) . $

If we use the same number of iterations $K_s = K$ at each scale, this reduces to the closed form upper bound $ norm(e_1^(K_1)) & lt.eq norm(e_S^0) d (K) (1 + d (K))^(S - 1)\
 & #h(2em) + med L_f / 2 abs(b - a) frac(d (K) (sqrt(2^S) (d (K) + 1)^S - sqrt(2) (d (K) + 1)), sqrt(2^S) (d (K) + 1) (sqrt(2) d (K) + sqrt(2) - 1)) $

where $d (K) = (1 - c)^K$ is a small number between $0 < d (K) < 1$.

#block[
#emph[Proof]. See #ref(<sec-freezed-multiscale-descent-error-proof>, supplement: [Section]).

]
] <thm-freezed-multiscale-descent-error>
We summarize the convergence with #ref(<cor-multiscale-convergence>, supplement: [Corollary]) that sends the number of iterations to infinity.

#corollary("Multiscale Convergence")[
As the total number of iterations grows $sum_(s = 1)^S K_s arrow.r oo$ (in the case of re-solve multi-scale in #ref(<thm-resolved-multiscale-descent-error>, supplement: [Theorem])) or the number of iterations at each scale $K_s arrow.r oo$ (in the case of freezed multi-scale in #ref(<thm-freezed-multiscale-descent-error>, supplement: [Theorem])), the final error goes to $0$, $lr(bar.v.double e_1^(K_1) bar.v.double)_2 arrow.r 0$, and we converge to the solution $x_1^(K_1) arrow.r x_1^(\*)$ at the finest scale.

] <cor-multiscale-convergence>
== Comparison With Projected Gradient Descent
<comparison-with-projected-gradient-descent>
From #ref(<prp-descent-lemma>, supplement: [Proposition]), projected gradient descent with a generic initialization on the fine grid gives us the following iterate convergence

$ norm(x_1^K - x_1^(\*)) lt.eq (1 - c)^K norm(x_1^0 - x_1^(\*)) . $

We can use this in combination with an expected initial error to get an expected number of iterations needed until we have converged to a desired tolerance. This is make precise by #ref(<thm-expected-pgd-convergence>, supplement: [Theorem]).

#theorem("Expected Projected Gradience Descent Convergence")[
Assume the problem is either scaled or shifted so that the solution is normalized $norm(x_1^(\*))_2 = 1$ or centred $norm(x_1^(\*))_2 = 0$ where $x_1^(\*) in bb(R)^I$. Choose an initialization $x_1^0 in bb(R)^I$ with i.i.d. standard normal entries $(x_1^0)_i tilde.op cal(N) (0 \, 1)$. Then the expected initial error is $ bb(E) norm(x_1^0 - x_1^(\*))_2 = sqrt(I + 1) . $

And, as the number of points grows $I arrow.r oo$, if we iterate projected gradient descent $K$ times where $ K gt.eq frac(log (1 \/ epsilon.alt) + log (I - 1) \/ 2, - log (1 - c)) \, $

the expected error is less than $epsilon.alt$,

$ bb(E) norm(x_1^0 - x_1^(\*))_2 lt.eq epsilon.alt . $

#block[
#emph[Proof]. See #ref(<sec-expected-pgd-convergence-proof>, supplement: [Section]).

]
] <thm-expected-pgd-convergence>
This is contrasted with the expected initial and final error for re-solved multi-scaled descent (#ref(<thm-expected-resolved-convergence>, supplement: [Theorem])) and freezed multi-scaled descent (#ref(<thm-expected-freezed-convergence>, supplement: [Theorem])).

#theorem("Expected Re-solved Multi-scale Convergence")[
Assume the same setting as in #ref(<thm-expected-pgd-convergence>, supplement: [Theorem]), but instead using re-solved multi-scaled descent method starting with a scale $s_0$. Let the number of points at the finest scale be $I = 2^S + 1$. Then we have the expected initial error

$ bb(E) norm(x_(s_0)^0 - x_(s_0)^(\*))_2 = sqrt(2^(S - s_0 + 1) + 2) . $

Performing $K_s$ iterations at each scale gives us the expected final error

$  & bb(E) norm(x_1^(K_1) - x_1^(\*))_2\
 & lt.eq (1 - c)^(K_1) sqrt(2^(S + 1)) ((1 - c)^(sum_(s = 2)^S K_s) + frac(L abs(a - b), 2 dot.op 2^S) + frac(L abs(a - b), 2 dot.op 2) sum_(s = 2)^(S - 1) (1 - c)^(sum_(t = 2)^s K_t) / 2^(S - s)) . $

#block[
#emph[Proof]. See #ref(<sec-expected-resolved-convergence-proof>, supplement: [Section]).

]
] <thm-expected-resolved-convergence>
It is not as straightforward to get the required number of iterations to achieve a desired level of accuracy. Additionally, this would not be a fair comparison with projected gradient descent since we expect an iteration at a coarse scale $S$ to be cheaper (less time and fewer floating point operations) than an iteration at the finest scale $s = 1$. For this reason, we need to cost an iteration of projected gradient descent at a scale $s$ in terms of the size of the problem at that scale.

#lemma("Cost of Projected Gradient Descent vs Multi-scale")[
Suppose the total cost of regular descent is given by $ C_(upright("GD")) = C_1 K $

where $C_1$ is the cost of performing one iteration at the finest scale $s = 1$.

The total cost of multi-scale descent is $ C_(upright("MS")) = sum_(s = 1)^S C_s K_s $

similarly.

If we assume the cost of projected gradient scales at least in the size of the problem ($C_1 = Omega (I)$, i.e.~$C_1 gt.eq C I$ for some $C gt.eq 0$), then the cost of projected gradient descent at scale $s$ is $ C_s gt.eq frac(2^(S - s + 1) + 1, 2^(S - s) + 1) C_(s + 1) gt.eq 3 / 2 C_(s + 1) \, $

which gives the total cost of multi-scale descent at most

$ C_(upright("MS")) lt.eq C_1 sum_(s = 1)^S (2 / 3)^(s - 1) K_s . $

#block[
#emph[Proof]. See #ref(<sec-cost-of-gd-vs-ms-proof>, supplement: [Section]).

]
] <lem-cost-of-gd-vs-ms>
We can use this to select a plan for the number of iterations at each scale $K_s$ that will be cheaper than projected gradient descent, yet still give the same upper bound on the expected final error.

#corollary("Sufficient Conditions for Re-solved Multi-scale to be Cheaper")[
Assume the problem is well conditioned with a condition number at least $c gt.eq 0.3$, and the finest scale problem is discretized with at least $I gt.eq 33 = 2^(4 + 1) + 1$ points. Then performing re-solved multi-scale with one iteration $K_s = 1$ at each scale except the finest scale where we iterate $K_1 = K - 3$ times, yields a tighter upper bound on the expected final error with a cheaper cost, than performing projected gradient descent with $K$ iterations.

#block[
#emph[Proof]. See #ref(<sec-resolved-cost-proof>, supplement: [Section]).

]
We can play a similar game for the freezed multi-scale approach.

] <cor-resolved-cost>
#theorem("Expected Freezed Multi-scale Convergence")[
Assumed the same setting as #ref(<thm-expected-pgd-convergence>, supplement: [Theorem]), but instead using the freezed multi-scale approach starting at the coarsest scale $S$, where the finest scale has $I = 2^S + 1$ many points. Then the expected initial error is two,

$ bb(E) norm(x_(s_0)^0 - x_(s_0)^(\*))_2 = 2 . $

Performing the same number of iterations $K_s = K$ at each scale gives us the expected final error

$  & bb(E) norm(x_1^K - x_1^(\*))_2\
 & lt.eq d (K) (2 (1 + d (K))^(S - 1) + L_f / 2 abs(b - a) frac((sqrt(2^S) (d (K) + 1)^S - sqrt(2) (d (K) + 1)), sqrt(2^S) (d (K) + 1) (sqrt(2) d (K) + sqrt(2) - 1))) $

where $d (K) = (1 - c)^K$, $c$ is the condition number for the problem, and the underlying continuous function $f$ is $L_f$ Lipschitz on the interval $[a \, b]$.

#block[
#emph[Proof]. See #ref(<sec-expected-freezed-convergence-proof>, supplement: [Section]).

]
] <thm-expected-freezed-convergence>
To better analyze when the bound for freezed multi-scale descent (#ref(<thm-expected-freezed-convergence>, supplement: [Theorem])) is small than for regular projected gradient descent (#ref(<thm-expected-pgd-convergence>, supplement: [Theorem])), we will look at the case when we have many iterations. This lets us approximate $d (K) approx 0$ since we know $d (K) arrow.r 0$ as $K arrow.r oo$.

#corollary("Sufficient Conditions for Freezed Multi-scale to be Cheaper")[
Assume the finest scale has at least $I gt.eq 2^S + 1$ many points where $S$ is at least

$ S > log_2 ((frac(L_f / 2 abs(b - a) + 1 \/ 2, (sqrt(2) - 1)) + 2)^2 - 2) . $

Then performing freezed multi-scale with $K_s = ceil.l K \/ 3 ceil.r - 1$ iterations at each scale starting with a scale $s = S$, yields a tighter upper bound on the expected final error with a cheaper cost, than performing projected gradient descent with $K$ iterations.

#block[
#emph[Proof]. See #ref(<sec-freezed-cost-proof>, supplement: [Section]).

]
] <cor-freezed-cost>
== Benchmarks
<benchmarks>
=== Synthetic Data
<synthetic-data>
TODO see multiscalesynthetic3d.jl

For a synthetic test, we generate 3 source distributions. Each distribution is a 3 dimensional product distribution of some standard continuous distributions.

```julia
using Distributions

source1a = Normal(4, 1)
source1b = Uniform(-7, 2)
source1c = Uniform(-1, 1)

source2a = Normal(0, 3)
source2b = Uniform(-2, 2)
source2c = Exponential(2)

source3a = Exponential(1)
source3b = Normal(0, 1)
source3c = Normal(0, 3)

source1 = product_distribution([source1a, source1b, source1c])
source2 = product_distribution([source2a, source2b, source2c])
source3 = product_distribution([source3a, source3b, source3c])

sources = (source1, source2, source3)
```

We generate the following $5 times 3$ mixing matrix

```julia
p1 = [0, 0.4, 0.6]
p2 = [0.3, 0.3, 0.4]
p3 = [0.8, 0.2, 0]
p4 = [0.2, 0.7, 0.1]
p5 = [0.6, 0.1, 0.3]

C_true = hcat(p1,p2,p3,p4,p5)'
```

and use it to construct $5$ mixture distributions.

```julia
distribution1 = MixtureModel([sources...], p1)
distribution2 = MixtureModel([sources...], p2)
distribution3 = MixtureModel([sources...], p3)
distribution4 = MixtureModel([sources...], p4)
distribution5 = MixtureModel([sources...], p5)
distributions = [distribution1, distribution2, distribution3, distribution4, distribution5]
```

These are discretized into $65 times 65 times 65$ sample tensors, and stacked into a $5 times 65 times 65 times 65$. We normalize the $1$-slices so that they sum to one.

```julia
sinks = [pdf.((d,), xyz) for d in distributions]
Y = cat(sinks...; dims=4)
# reorder so the first dimension indexes mixtures rather than the final dimension
Y = permutedims(Y, (4,1,2,3))
Y_slices = eachslice(Y, dims=1)
correction = sum.(Y_slices) # normalize slices to 1
Y_slices ./= correction
```

Because these are large tensors, we only test a single shot decomposition (after they have been compiled). This gives us the following.

```julia
# Run once to compile functions
factorize(Y; options...);
multiscale_factorize(Y; continuous_dims=[2, 3, 4], options...);
```

```julia
# Time the functions
@time decomposition, stats_data, kwargs = factorize(Y; options...);
```

```shell
11.828295 seconds (214.97 k allocations: 15.197 GiB, 32.42% gc time, 0.00% compilation time)
```

```julia
@time decomposition, stats_data, kwargs = multiscale_factorize(Y; continuous_dims=[2, 3, 4], options...);
```

```shell
2.335374 seconds (201.33 k allocations: 2.905 GiB, 25.26% gc time, 0.00% compilation time)
```

We can see that `multiscale_factorize` is roughly five times as fast and uses about a fifth of the memory in this example.

=== Real Data
<real-data>
We use the same sedimentary data and Tucker-$1$ model as described in @graham_tracing_2025 to factorize a tensor containing mixtures of estimated densities. We discretize the densities with $K = 2^10 + 1 = 1025$ points to obtain an input tensor $Y in bb(R)_(+)^(20 times 7 times 1025)$ and normalize the depth fibres so that $sum_(k in [K]) Y [i \, j \, k] = 1$ for all $i in [20]$ and $j in [7]$.

We run the multi-scale factorization algorithm

```julia
multiscale_factorize(Y; continuous_dims=3, options...)
```

with the following options.

```julia
options = (
    rank=3,
    momentum=false,
    do_subblock_updates=false,
    model=Tucker1,
    tolerance=(0.12),
    converged=(RelativeError), # relative error ≤ 12%
    constrain_init=true,
    constraints=[l1scale_average12slices! ∘ nonnegative!, nnonnegative!],
    stats=[Iteration, ObjectiveValue, GradientNNCone, RelativeError],
    maxiter=200
)
```

We use the same convergence criteria at each scale and iterate until the relative error between the input $Y$ and our model $X$ is at most $12 %$, or until $200$ iterations have passed. We use $12 %$ because this is roughly the error Graham et. al.~observe in their final factorization @graham_tracing_2025. The third dimension is specified as continuous since each depth fibre $Y [i \, j \, :]$ is a discretized continuous probability density function.

This is compared to the regular factorization algorithm

```julia
factorize(Y; options...)
```

using the Julia package `BenchmarkingTools`. This runs the algorithm as many times as it can within a default time window. Note that a new random initialization is generated for each run. After running the following,

```julia
using BenchmarkingTools

# Run each function once so they compile
factorize(Y; options...);
multiscale_factorize(Y; continuous_dims=3,options...);

benchmark1 = @benchmark factorize(Y; options...)
display(benchmark1)

benchmark2 = @benchmark multiscale_factorize(Y; continuous_dims=3,options...)
display(benchmark2)
```

we observe the two benchmarks. For `factorize`, we have

```julia
BenchmarkTools.Trial: 22 samples with 1 evaluation per sample.
Range (min … max):  103.648 ms … 541.438 ms  ┊ GC (min … max): 13.87% … 14.20%
Time  (median):     191.769 ms               ┊ GC (median):    15.00%
Time  (mean ± σ):   227.315 ms ± 130.666 ms  ┊ GC (mean ± σ):  15.04% ±  1.79%

▃▃     █    ▃    ▃
██▇▁▇▁▁█▁▁▇▇█▇▁▁▁█▁▁▁▇▁▁▇▁▁▁▁▁▁▁▁▁▁▇▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▇▁▇▁▁▁▁▇ ▁
104 ms           Histogram: frequency by time          541 ms <

Memory estimate: 201.53 MiB, allocs estimate: 78879.
```

and for `multiscale_factorize`, we observe the following.

```julia
BenchmarkTools.Trial: 59 samples with 1 evaluation per sample.
Range (min … max):  63.974 ms … 147.756 ms  ┊ GC (min … max):  0.00% … 22.20%
Time  (median):     81.719 ms               ┊ GC (median):    13.13%
Time  (mean ± σ):   85.094 ms ±  13.163 ms  ┊ GC (mean ± σ):  12.13% ±  5.42%

            ▁█  ▁ ▁      ▁
▆▁▁▁▁▁▁▁▆▁▄▆▇██▆▇█▇█▄▆▄▄▄▆█▁▁▁▄▁▄▄▁▄▁▁▄▄▁▁▁▁▁▁▄▁▁▁▁▁▁▁▁▁▁▁▁▄ ▁
64 ms           Histogram: frequency by time          125 ms <

Memory estimate: 84.93 MiB, allocs estimate: 634376.
```

By every metric, the multi-scale approach is faster and uses less memory than the regular factorize approach. Looking at the median time and memory estimate, it is roughly twice as fast, and uses about half as much memory in the case of this problem. These wall clock times include a nontrivial amount of garbage collection (GC) so it means there could be design improvements such as using more preallocated arrays to reduce memory usage. But these improvements would speed up both `factorize` and `multiscale_factorize`. Without these improvements, we can see that `multiscale_factorize` is less likely to use garbage collection since more computations are occurring on smaller arrays.

= Conclusion
<conclusion>
The BlockTensorFactorization.jl Julia provides a new all-in-one package for performing constrained tensor factorizations, and a playground for designing new decompositions and custom constraints. Careful design elements were engineered to balance flexibility and efficiency. By creating this package, new advancements like enforcing constraints through scaling rather than projection, and performing optimization over multiple scales were mathematically examined and numerically tested. These novel ideas are worth investigating further to see their applicability to continuous optimization beyond tensor factorization.

= Appendix
<appendix>
== Building the Hessian from two gradients
<sec-hessian-from-gradient>
To build the Hessian from the definition of the gradient, we first extend the gradient to tensor-valued functions. For a function $F : bb(R)^(J_1 times dots.h.c times J_N) arrow.r bb(R)^(I_1 times dots.h.c times I_M)$ where

$ F (X) = [f_(i_1 dots.h i_M) (X)] $

is a tensor of scalar functions $f_(i_1 dots.h i_M) : T arrow.r bb(R)$, the gradient of $F$ at $X$ is defined entry-wise as

$ nabla F (X) [i_1 \, dots.h \, i_M] [j_1 \, dots.h \, j_N] & = nabla f_(i_1 dots.h i_M) (X) [j_1 \, dots.h \, j_N]\
 & = frac(partial f_(i_1 dots.h i_M), partial X [j_1 \, dots.h \, j_N]) (X) . $

This treats the gradient at $X$ as a tensor of tensors $nabla F (X) in (bb(R)^(I_1 times dots.h.c times I_M))^(J_1 times dots.h.c times J_N)$. This is naturally isomorphic to a tensor of order $M + N$ with entries

#math.equation(block: true, numbering: "(1)", [ $ nabla F (X) [i_1 \, dots.h \, i_M \, j_1 \, dots.h \, j_N] = frac(partial f_(i_1 dots.h i_M), partial X [j_1 \, dots.h \, j_N]) (X) . $ ])<eq-tensor-gradient-entries>

So we conclude that the gradient at $X$ of a tensor-valued function $F$ is $nabla F : bb(R)^(J_1 times dots.h.c times J_N) arrow.r bb(R)^(I_1 times dots.h.c times I_M times J_1 times dots.h.c times J_N)$ is given by #ref(<eq-tensor-gradient-entries>, supplement: [Equation]).

We can define the Hessian of a scalar function $f : bb(R)^(I_1 times dots.h.c times I_N) arrow.r bb(R)$ at $X$ as $nabla^2 f (X) = nabla (nabla f) (X) : bb(R)^(I_1 times dots.h.c times I_N) arrow.r bb(R)^((I_1 times dots.h.c times I_N)^2)$. The inner nabla $nabla$ is the gradient of the scalar function $f$, and the outer nabla $nabla$ is the gradient of the tensor-valued function $nabla f$.

This means

$ nabla^2 f (A) [i_1 \, dots.h \, i_N \, j_1 \, dots.h \, j_N] = frac(partial^2 f, partial A [j_1 \, dots.h \, j_N] partial A [i_1 \, dots.h \, i_N]) (A) \, $

but if the function has continuous second derivatives, we can perform the partial derivatives in either order

$ frac(partial^2 f, partial A [j_1 \, dots.h \, j_N] partial A [i_1 \, dots.h \, i_N]) (A) = frac(partial^2 f, partial A [i_1 \, dots.h \, i_N] partial A [j_1 \, dots.h \, j_N]) (A) . $

== Randomizing the order of updates
<randomizing-the-order-of-updates>
#figure([
#table(
  columns: (20%, 15%, 27%, 37%),
  align: (auto,auto,auto,auto,),
  table.header([`group_by_factor`], [`random_order`], [`recursive_random_order`], [Description],),
  table.hline(),
  [`false`], [`false`], [`false`], [In the order given],
  [`false`], [`false`], [`true`], [In order given, but randomize how existing blocks are ordered (recursively)],
  [`false`], [`true`], [`false`], [Randomize updates, but keep existing blocks in order],
  [`false`], [`true`], [`true`], [Fully random],
  [`true`], [`false`], [`false`], [In the order given],
  [`true`], [`false`], [`true`], [In order of factors, but updates for each factor a random order],
  [`true`], [`true`], [`false`], [Random order of factors, preserve order of updates within each factor],
  [`true`], [`true`], [`true`], [Almost fully random, but updates for each factor are done together],
)
], caption: figure.caption(
position: top, 
[
Full description of randomizing the order of updates within a `BlockUpdate`.
]), 
kind: "quarto-float-tbl", 
supplement: "Table", 
)
<tbl-blockupdate-randomization>


== Constraint Rescaling Proofs
<constraint-rescaling-proofs>
In the following proofs, we have an $L_f$-Lipschitz function $f : [l \, u] arrow.r bb(R)$ that is discretized according to

$ x [i] = f (t [i]) \, $

where

$ t [i] = l + (i - 1) Delta t = l + (i - 1) frac(u - l, I - 1) . $

See #ref(<thm-rem-constraint-assumptions>, supplement: [Theorem]).

This ensures neighboring entries of $x$ are close together. Specifically, we have

$ abs(x_i - x_(i + 1)) = abs(f (t_i) - f (t_(i + 1))) lt.eq L_f abs(t_i - t_(i + 1)) = L_f frac(u - l, I - 1) := C_x . $

=== Linear Constraint Scaling
<sec-linear-constraint-scaling-proof>
#block[
#emph[Proof]. First, assume an even number of points $I$. $ abs(2 ⟨overline(a) \, overline(x)⟩ - b) & = abs(2 sum_(i med upright("odd")) a_i x_i - sum_(i = 1)^I a_i x_i)\
 & = abs(sum_(i med upright("odd")) a_i x_i - sum_(i med upright("even")) a_i x_i)\
 & = abs(sum_(i med upright("odd")) a_i x_i - sum_(i med upright("odd")) a_(i + 1) x_(i + 1))\
 & lt.eq sum_(i med upright("odd")) abs(a_i x_i - a_(i + 1) x_(i + 1))\
 & = sum_(i med upright("odd")) abs(a_i x_i - a_(i + 1) x_i + a_(i + 1) x_i - a_(i + 1) x_(i + 1))\
 & lt.eq sum_(i med upright("odd")) (abs(a_i x_i - a_(i + 1) x_i) + abs(a_(i + 1) x_i - a_(i + 1) x_(i + 1)))\
 & = sum_(i med upright("odd")) (abs(x_i) abs(a_i - a_(i + 1)) + abs(a_(i + 1)) abs(x_i - x_(i + 1)))\
 & lt.eq max (lr(bar.v.double a bar.v.double)_oo \, lr(bar.v.double x bar.v.double)_oo) sum_(i med upright("odd")) (abs(a_i - a_(i + 1)) + abs(x_i - x_(i + 1)))\
 & lt.eq max (lr(bar.v.double a bar.v.double)_oo \, lr(bar.v.double x bar.v.double)_oo) sum_(i med upright("odd")) (C_a + C_x)\
 & = max (lr(bar.v.double a bar.v.double)_oo \, lr(bar.v.double x bar.v.double)_oo) (C_a + C_x) I / 2\
 & = max (lr(bar.v.double a bar.v.double)_oo \, lr(bar.v.double x bar.v.double)_oo) frac(I (L_g + L_f) (u - l), 2 (I - 1))\
 & lt.eq max (lr(bar.v.double g bar.v.double)_oo \, lr(bar.v.double f bar.v.double)_oo) frac(I (L_g + L_f) (u - l), 2 (I - 1)) $

]
=== $L_1$ norm Constraint
<sec-l1-norm-constraint-scaling-proof>
#block[
#emph[Proof]. For $I$ even, $ abs(2 norm(overline(x)) - lr(bar.v.double x bar.v.double)_1) & = abs(2 sum_(i med upright("odd")) lr(|x_i) - sum_(i = 1)^I abs(x_i)|)\
 & = abs(sum_(i med upright("odd")) lr(|x_i) + sum_(i med upright("odd")) abs(x_i) - (sum_(i med upright("odd")) abs(x_i) + sum_(i med upright("even")) abs(x_i))|)\
 & = abs(sum_(i med upright("odd")) lr(|x_i) - sum_(i med upright("even")) abs(x_i)|)\
 & = abs(sum_(i med upright("odd")) lr(|x_i) - sum_(i med upright("odd")) abs(x_(i + 1))|)\
 & lt.eq sum_(i med upright("odd")) abs(lr(|x_i) - abs(x_(i + 1))|)\
 & lt.eq sum_(i med upright("odd")) abs(x_i - x_(i + 1))\
 & lt.eq sum_(i med upright("odd")) C\
 & = C I / 2\
 & = frac(I L (u - l), 2 (I - 1)) . $

For $I$ odd, we have $ abs(2 norm(overline(x)) - frac(I + 1, I) lr(bar.v.double x bar.v.double)_1) & = abs(2 norm(overline(x)) - lr(bar.v.double x bar.v.double)_1 - lr(bar.v.double x bar.v.double)_1 / I)\
 & = abs(2 sum_(i med upright("odd")) lr(|x_i) - sum_(i = 1)^I abs(x_i) - lr(bar.v.double x bar.v.double)_1 / I|)\
 & = abs(sum_(i med upright("odd")) lr(|x_i) + sum_(i med upright("odd")) abs(x_i) - (sum_(i med upright("odd")) abs(x_i) + sum_(i med upright("even")) abs(x_i)) - lr(bar.v.double x bar.v.double)_1 / I|)\
 & = abs(sum_(i med upright("odd")) lr(|x_i) - sum_(i med upright("even")) abs(x_i) - lr(bar.v.double x bar.v.double)_1 / I|)\
 & = abs(sum_(j = 1)^((I - 1) \/ 2) lr(|x_(2 j - 1)) + abs(x_I) - sum_(j = 1)^((I - 1) \/ 2) abs(x_(2 j)) - lr(bar.v.double x bar.v.double)_1 / I|)\
 & lt.eq sum_(j = 1)^((I - 1) \/ 2) abs(lr(|x_(2 j - 1)) - abs(x_(2 j))|) + abs(lr(|x_I) - lr(bar.v.double x bar.v.double)_1 / I|)\
 & lt.eq sum_(j = 1)^((I - 1) \/ 2) abs(x_(2 j - 1) - x_(2 j)) + 1 / I abs(I lr(|x_I) - lr(bar.v.double x bar.v.double)_1|)\
 & lt.eq sum_(j = 1)^((I - 1) \/ 2) C + 1 / I abs(sum_(i = 1)^I (lr(|x_I) - abs(x_i))|)\
 & lt.eq C frac(I - 1, 2) + 1 / I sum_(i = 1)^I abs(lr(|x_I) - abs(x_i)|)\
 & lt.eq C frac(I - 1, 2) + 1 / I sum_(i = 1)^I abs(x_I - x_i)\
 & lt.eq C frac(I - 1, 2) + 1 / I sum_(i = 1)^I C (I - i) med upright("(Apply Lipschitz recursively)")\
 & = C frac(I - 1, 2) + 1 / I frac(C I (I - 1), 2)\
 & = frac(L (u - l), 2) + frac(L (u - l), 2)\
 & = L (u - l) . $

]
== Multi-scale Convergence Proofs
<multi-scale-convergence-proofs>
=== Inexact Interpolation Error Proof
<sec-inexact-interpolation-proof>
Proof of #ref(<lem-inexact-interpolation>, supplement: [Lemma]).

#emph[Proof]. We let the inexact values be $tilde(y) [k] = y [k] + delta_k$, and we interpolate the inexact values to get $hat(tilde(Y))$: $ hat(tilde(Y)) [j] & = cases(delim: "{", tilde(y) [frac(j + 1, 2)] & upright("if ") j upright(" is odd"), 1 / 2 (tilde(y) [j / 2] + tilde(y) [j / 2 + 1]) & upright("if ") j upright(" is even"))\
 & = cases(delim: "{", y [frac(j + 1, 2)] + delta_(frac(j + 1, 2)) & upright("if ") j upright(" is odd"), 1 / 2 (y [j / 2] + y [j / 2 + 1]) + 1 / 2 (delta_(j / 2) + delta_(j / 2 + 1)) & upright("if ") j upright(" is even")) . $

So for odd $j$, $ abs(hat(tilde(Y)) [j] - hat(Y) [j]) = abs(delta_(frac(j + 1, 2))) := e [j] \, $

and even $j$, $ abs(hat(tilde(Y)) [j] - hat(Y) [j]) = 1 / 2 abs(delta_(j / 2) + delta_(j / 2 + 1)) := e [j] . $

So we have $ norm(hat(tilde(Y)) - hat(Y)) = sqrt(sum_(j in [J]) abs(hat(tilde(Y)) [j] - hat(Y) [j])^2) = sqrt(sum_(j in [J]) abs(e [j])^2) = lr(bar.v.double e bar.v.double) $

We would like some bound on $lr(bar.v.double e bar.v.double)$ in terms of the closeness of $y$ and $tilde(y)$ (that is, in terms of $delta$).

$ lr(bar.v.double e bar.v.double)^2 & = sum_(j med upright("odd")) abs(e [j])^2 + sum_(j med upright("even")) abs(e [j])^2\
 & = sum_(j med upright("odd")) abs(delta_(frac(j + 1, 2)))^2 + sum_(j med upright("even")) abs(1 / 2 (delta_(j / 2) + delta_(j / 2 + 1)))^2\
 & = sum_(k = 1)^K abs(delta_k)^2 + 1 / 4 sum_(k = 1)^(K - 1) (delta_k + delta_(k + 1))^2\
 & = norm(delta)^2 + 1 / 4 sum_(k = 1)^(K - 1) (delta_k^2 + delta_(k + 1)^2 + 2 delta_k delta_(k + 1))\
 & = norm(delta)^2 + 1 / 4 (sum_(k = 1)^(K - 1) delta_k^2 + sum_(k = 1)^(K - 1) delta_(k + 1)^2 + 2 sum_(k = 1)^(K - 1) delta_k delta_(k + 1))\
 & lt.eq norm(delta)^2 + 1 / 4 (sum_(k = 1)^K delta_k^2 + sum_(k = 0)^(K - 1) delta_(k + 1)^2 + 2 sum_(k = 1)^(K - 1) delta_k delta_(k + 1))\
 & = norm(delta)^2 + 1 / 4 (norm(delta)^2 + norm(delta)^2 + 2 sum_(k = 1)^(K - 1) delta_k delta_(k + 1))\
 & = 3 / 2 norm(delta)^2 + 1 / 2 sum_(k = 1)^(K - 1) delta_k delta_(k + 1)\
 & lt.eq 3 / 2 norm(delta)^2 + 1 / 2 sqrt(sum_(k = 1)^(K - 1) delta_k^2) sqrt(sum_(k = 1)^(K - 1) delta_(k + 1)^2) quad upright("(Cauchy–Schwarz)")\
 & lt.eq 3 / 2 norm(delta)^2 + 1 / 2 sqrt(sum_(k = 1)^K delta_k^2) sqrt(sum_(k = 0)^(K - 1) delta_(k + 1)^2)\
 & = 3 / 2 norm(delta)^2 + 1 / 2 norm(delta) norm(delta)\
 & = 2 norm(delta)^2 $

Therefore, $ lr(bar.v.double e bar.v.double) lt.eq sqrt(2) norm(delta) $

or substituting our notation, $ norm(hat(tilde(Y)) - hat(Y)) lt.eq sqrt(2) norm(tilde(y) - y) . $

This is saying that the error in our fine grid $(Y)$ is bounded by a factor of $sqrt(2)$ of the error in the coarse grid $(y)$ when we double the number of points.

Now we can use the triangle inequality to bound the difference between the interpolated approximate values $hat(tilde(Y))$ and the true values on the finer grid $Y$: $ norm(hat(tilde(Y)) - Y) & lt.eq norm(hat(tilde(Y)) - hat(Y)) + norm(hat(Y) - Y)\
 & lt.eq sqrt(2) norm(tilde(y) - y) + frac(L, 2 sqrt(K - 1)) norm(a - b)_2 . $

END OF PROOF

=== Re-solved Multi-scale Descent Error Proof
<sec-resolved-multiscale-descent-error-proof>
Proof of #ref(<thm-resolved-multiscale-descent-error>, supplement: [Theorem]).

Using the lemmas in #ref(<sec-analysis-lemmas>, supplement: [Section]), we showed $ norm(hat(tilde(Y)) - Y) lt.eq sqrt(2) norm(tilde(y) - y) + frac(L abs(a - b), 2 sqrt(K - 1)) . $ Note the $L$ is the Lipchitz constant of the continuous function $f$, not the smoothness of the objective $cal(L)$ above, and the $K$ is the number of points in the discretization, not the number of iterations. Translating this into our notation for the multi-scaled descent, we have $ norm(x_(s / 2)^0 - x_(s / 2)^(\*)) = norm(hat(x)_s^(K_s) - x_(s / 2)^(\*)) lt.eq sqrt(2) norm(x_s^(K_s) - x_s^(\*)) + frac(L abs(a - b), 2 sqrt(2^(S - s + 1))) . $ Note that at scale $s$, we have $2^(S - s + 1) + 1$ many points in our discretization, where we have $2^S + 1$ many points in our finest scale, and the discretization is over the interval $a lt.eq t lt.eq b$ for an $L$ Lipschitz function $f$.

Combining this with our convergence for gradient descent gives us the inequality $ norm(x_(s / 2)^0 - x_(s / 2)^(\*)) & lt.eq sqrt(2) norm(x_s^(K_s) - x_s^(\*)) + frac(L abs(a - b), 2 sqrt(2^(S - s + 1)))\
 & lt.eq sqrt(2) (1 - c)^(K_s) norm(x_s^0 - x_s^(\*)) + frac(L abs(a - b), 2 sqrt(2^(S - s + 1))) $ I noticed I am using $s$ in two different ways here. Originally, $s$ was supposed to be the number of points we skip, as in "only keep every $s$ points in our discretization". This is meant to start at some large power of $2$ and keep halving until we hit $1$. But the $s$ in $2^(S - s + 1)$ counts what scale we are at and starts at $S$ and decreasing by $1$ until we hit $1$. The second definition makes more sense so we will go with that from now on. This means our descent lemma looks like $ norm(x_(s - 1)^0 - x_(s - 1)^(\*)) & lt.eq sqrt(2) (1 - c)^(K_s) norm(x_s^0 - x_s^(\*)) + frac(L abs(a - b), 2 sqrt(2^(S - s + 1))) . $ Or writing it in terms of $s + 1$: $ norm(x_s^0 - x_s^(\*)) & lt.eq sqrt(2) (1 - c)^(K_(s + 1)) norm(x_(s + 1)^0 - x_(s + 1)^(\*)) + sqrt(2^s) frac(L abs(a - b), 2 sqrt(2^S)) . $ Let $e_s = norm(x_s^0 - x_s^(\*))$ and $C = frac(L abs(a - b), 2 sqrt(2^(S + 1)))$ to clean up notation

$ e_s & lt.eq sqrt(2) (1 - c)^(K_(s + 1)) e_(s + 1) + sqrt(2^(s + 1)) C . $ Now we recurse! Starting from the last initial error at the finest scale $e_1$, we want to write this in terms of the first initial error at the largest scale $e_S$. $ e_1 & lt.eq sqrt(2) (1 - c)^(K_2) e_2 + sqrt(2^2) C\
 & lt.eq sqrt(2) (1 - c)^(K_2) (sqrt(2) (1 - c)^(K_3) e_3 + sqrt(2^3) C) + sqrt(2^2) C\
 & = sqrt(2) (1 - c)^(K_2) sqrt(2) (1 - c)^(K_3) e_3 + sqrt(2) (1 - c)^(K_2) sqrt(2^3) C + sqrt(2^2) C\
 & = sqrt(2)^2 (1 - c)^(K_2 + K_3) e_3 + sqrt(2) (1 - c)^(K_2) sqrt(2^3) C + sqrt(2^2) C $ Before going further, we actually want to run gradient descent with the starting point $x_1^0$ (the $x$ inside $e_1$), so we should really be writing $  & norm(x_1^(K_1) - x_1^(\*))\
 & lt.eq (1 - c)^(K_1) norm(x_1^0 - x_1^(\*))\
 & = (1 - c)^(K_1) e_1\
 & lt.eq (1 - c)^(K_1) (sqrt(2) (1 - c)^(K_2) e_2 + sqrt(2^2) C)\
 & = sqrt(2) (1 - c)^(K_1 + K_2) e_2 + (1 - c)^(K_1) sqrt(2^2) C\
 & lt.eq sqrt(2) (1 - c)^(K_1 + K_2) (sqrt(2) (1 - c)^(K_3) e_3 + sqrt(2^3) C) + (1 - c)^(K_1) sqrt(2^2) C\
 & = sqrt(2)^2 (1 - c)^(K_1 + K_2 + K_3) e_3 + sqrt(2) (1 - c)^(K_1 + K_2) sqrt(2^3) C + (1 - c)^(K_1) sqrt(2^2) C\
 & lt.eq sqrt(2)^2 (1 - c)^(K_1 + K_2 + K_3) (sqrt(2) (1 - c)^(K_4) e_4 + sqrt(2^4) C) + sqrt(2) (1 - c)^(K_1 + K_2) sqrt(2^3) C + (1 - c)^(K_1) sqrt(2^2) C\
 & = sqrt(2)^3 (1 - c)^(K_1 + K_2 + K_3 + K_4) e_4 + sqrt(2)^2 (1 - c)^(K_1 + K_2 + K_3) sqrt(2^4) C + sqrt(2) (1 - c)^(K_1 + K_2) sqrt(2^3) C + (1 - c)^(K_1) sqrt(2^2) C\
 & med med dots.v\
 & lt.eq sqrt(2)^(S - 1) (1 - c)^(sum_(s = 1)^S K_s) e_S + sum_(s = 1)^(S - 1) sqrt(2)^(s - 1) (1 - c)^(sum_(t = 1)^s K_t) sqrt(2^(s + 1)) C\
 & = sqrt(2^(S - 1)) (1 - c)^(sum_(s = 1)^S K_s) e_S + C sum_(s = 1)^(S - 1) 2^s (1 - c)^(sum_(t = 1)^s K_t) . $

=== Freezed Multi-scale Descent Error Proof
<sec-freezed-multiscale-descent-error-proof>
Proof of #ref(<thm-freezed-multiscale-descent-error>, supplement: [Theorem]).

Using the work below, we have $ norm(e_((s))^0) lt.eq frac(L abs(b - a), 2 sqrt(2^(S - s))) + norm(e_(s + 1)^(K_(s + 1))) . $ Putting this back into the GD inequality gets us $ norm(e_((s))^(K_s)) lt.eq (1 - c)^(K_s) (frac(L abs(b - a), 2 sqrt(2^(S - s))) + norm(e_(s + 1)^(K_(s + 1)))) . $ And finally, we get the recursion relation $ norm(e_s^(K_s))^2 & lt.eq (1 - c)^(2 K_s) (frac(L abs(b - a), 2 sqrt(2^(S - s))) + norm(e_(s + 1)^(K_(s + 1))))^2 + norm(e_(s + 1)^(K_(s + 1)))^2 $ We will use big $C = L / 2 abs(b - a)$ to clean up a bit. Note little $c = mu / L_(nabla ell)$ is the ratio of strongly convex to smoothness of the loss function $ell$. $ norm(e_s^(K_s))^2 & lt.eq (1 - c)^(2 K_s) (C / sqrt(2^(S - s)) + norm(e_(s + 1)^(K_(s + 1))))^2 + norm(e_(s + 1)^(K_(s + 1)))^2 $ Or the looser bound by removing the squares $ norm(e_s^(K_s)) & lt.eq (1 - c)^(K_s) (C / sqrt(2^(S - s)) + norm(e_(s + 1)^(K_(s + 1)))) + norm(e_(s + 1)^(K_(s + 1)))\
 & = (1 + (1 - c)^(K_s)) norm(e_(s + 1)^(K_(s + 1))) + C (1 - c)^(K_s) 1 / sqrt(2^(S - s)) $

- we have true points $x$ which are true sample values of an $L$ Lipschitz function
- we have an approximation of $x$, given by $hat(x)$
- then we have a linear interpolation of $hat(x)$ given by $hat(X)$. These are ONLY the in-between points
- the true values at the in-between points are $Y$
- the linear interpolation of the true points is $X$
- the respective errors are $e$ and $E$
- want to bound $E$ by $e$
- suppose we have $M = 2^(S - s) + 1$ points at the coarse scale for $x$, for some positive integer $s$ (this is the scale $s + 1$, the previous scale)
- we will have $N = 2^(S - s) = M - 1$ in-between points at the finer grid for $X$
  - Note, this gives $2^(S - s + 1) + 1$ may points at the current scale $s$

We have $ hat(x)_i - x_i = e_i $ and the linear interpolation (only the in-between points) is $ hat(X)_j = 1 / 2 (hat(x)_j + hat(x)_(j + 1)) $ Note there is one fewer point. Similarly, with the interpolation of the true values $ X_j = 1 / 2 (x_j + x_(j + 1)) $ The error is defined as $ E_j = hat(X)_j - Y_j . $ This is what we want to bound. We also have the difference between the interpolation of the approximate and interpolation of the true values $ hat(X)_j - X_j = 1 / 2 (e_j + e_(j + 1)) := delta_j $

There is also the difference between the true values at the in-between points $Y$ and the interpolation of the true points $X$. $ X_j - Y_j $ We have $ norm(E) = norm(hat(X) - Y) lt.eq norm(X - Y) + norm(hat(X) - X) = norm(X - Y) + norm(delta) $ Now $ norm(X - Y)^2 & = sum_(j = 1)^N (X_j - Y_j)^2\
 & lt.eq sum_(j = 1)^N (L / 2 Delta_j)^2\
 & lt.eq N (L / 2 Delta_j)^2 $ where $Delta_i$ is the (input) space between the $i$ and $i + 1$ points. For $M$ equally spaced points on (and including the boundary) the interval $[a \, b]$, $ Delta_j = frac(b - a, M - 1) = frac(b - a, N) $ So we have $ norm(X - Y) lt.eq sqrt(N) L / 2 frac(b - a, N) = frac(L abs(b - a), 2 sqrt(N)) . $ Now we look at the second term $delta$. $ norm(delta)^2 & = sum_(j = 1)^N delta_j^2\
 & = sum_(j = 1)^N 1 / 2^2 (e_j + e_(j + 1))^2\
 & = 1 / 4 sum_(j = 1)^N (e_j^2 + e_(j + 1)^2 + 2 e_j e_(j + 1))\
 & = 1 / 4 (sum_(j = 1)^N e_j^2 + sum_(j = 1)^N e_(j + 1)^2 + 2 sum_(j = 1)^N e_j e_(j + 1))\
 & lt.eq 1 / 4 (sum_(j = 1)^N e_j^2 + sum_(j = 1)^N e_(j + 1)^2 + 2 sqrt(sum_(j = 1)^N e_j^2) sqrt(sum_(j = 1)^N e_(j + 1)^2))\
 & lt.eq 1 / 4 (sum_(i = 1)^M e_i^2 + sum_(i = 1)^M e_j^2 + 2 sqrt(sum_(i = 1)^M e_i^2) sqrt(sum_(i = 1)^M e_i^2))\
 & = 1 / 4 (norm(e)^2 + norm(e)^2 + 2 norm(e) norm(e))\
 & = 1 / 4 (4 norm(e)^2)\
 & = lr(bar.v.double e bar.v.double)^2\
 $

So finally, we have $ norm(E) & lt.eq frac(L abs(b - a), 2 sqrt(N)) + lr(bar.v.double e bar.v.double) . $ This is the same thing as we got in the previous attempt, but without the factor of $sqrt(2)$.

We have the relation $ norm(e_s^(K_s)) & lt.eq (1 + (1 - c)^(K_s)) norm(e_(s + 1)^(K_(s + 1))) + C (1 - c)^(K_s) 1 / sqrt(2^(S - s)) $ for every $s = 1 \, dots.h \, S$. So let's expand this. $ norm(e_1^(K_1)) & lt.eq (1 + (1 - c)^(K_1)) norm(e_2^(K_2)) + C (1 - c)^(K_1) 1 / sqrt(2^(S - 1))\
 & lt.eq (1 + (1 - c)^(K_1)) ((1 + (1 - c)^(K_2)) norm(e_3^(K_3)) + C (1 - c)^(K_2) 1 / sqrt(2^(S - 2))) + C (1 - c)^(K_1) 1 / sqrt(2^(S - 1))\
 & = (1 + (1 - c)^(K_1)) (1 + (1 - c)^(K_2)) norm(e_3^(K_3))\
 & #h(2em) + med C (1 + (1 - c)^(K_1)) (1 - c)^(K_2) 1 / sqrt(2^(S - 2)) + C (1 - c)^(K_1) 1 / sqrt(2^(S - 1))\
 & lt.eq (1 + (1 - c)^(K_1)) (1 + (1 - c)^(K_2)) ((1 + (1 - c)^(K_3)) norm(e_4^(K_4)) + C (1 - c)^(K_3) 1 / sqrt(2^(S - 3)))\
 & #h(2em) + med C (1 + (1 - c)^(K_1)) (1 - c)^(K_2) 1 / sqrt(2^(S - 2)) + C (1 - c)^(K_1) 1 / sqrt(2^(S - 1))\
 & = (1 + (1 - c)^(K_1)) (1 + (1 - c)^(K_2)) (1 + (1 - c)^(K_3)) norm(e_4^(K_4))\
 & #h(2em) + med C (1 + (1 - c)^(K_1)) (1 + (1 - c)^(K_2)) (1 - c)^(K_3) 1 / sqrt(2^(S - 3))\
 & #h(2em) + med C (1 + (1 - c)^(K_1)) (1 - c)^(K_2) 1 / sqrt(2^(S - 2)) + C (1 - c)^(K_1) 1 / sqrt(2^(S - 1)) $ So the general formula goes to $ norm(e_1^(K_1)) & lt.eq norm(e_S^(K_S)) product_(s = 1)^(S - 1) (1 + (1 - c)^(K_s)) + C sum_(s = 1)^(S - 1) 1 / sqrt(2^(S - s)) (1 - c)^(K_s) product_(j = 1)^(s - 1) (1 + (1 - c)^(K_j)) . $ After adding the GD on the coarsest scale, we get $ norm(e_1^(K_1)) & lt.eq norm(e_S^0) (1 - c)^(K_S) product_(s = 1)^(S - 1) (1 + (1 - c)^(K_s)) + C sum_(s = 1)^(S - 1) 1 / sqrt(2^(S - s)) (1 - c)^(K_s) product_(j = 1)^(s - 1) (1 + (1 - c)^(K_j)) . $

This gives us the first upper bound for any plan of iterations $K_s$. If we now assume each scale uses the name number of iterations $K_s = K$ we get the following.

$ norm(e_1^(K_1)) & lt.eq norm(e_S^0) (1 - c)^K (1 + (1 - c)^K)^(S - 1) + C sum_(s = 1)^(S - 1) 1 / sqrt(2^(S - s)) (1 - c)^K (1 + (1 - c)^K)^(s - 1) . $

Via wolfram alpha, we have

$  & sum_(s = 1)^(S - 1) ((1 - c)^K ((1 - c)^K + 1)^(s - 1)) 1 / sqrt(2^(S - s))\
 & = frac((1 - c)^K (sqrt(2^S) ((1 - c)^K + 1)^S - sqrt(2) ((1 - c)^K + 1)), sqrt(2^S) ((1 - c)^K + 1) (sqrt(2) (1 - c)^K + sqrt(2) - 1)) $

Setting $d (K) = (1 - c)^K$ gives the final upper bound in the theorem.

=== Expected Projected Gradient Descent Convergence Proof
<sec-expected-pgd-convergence-proof>
Proof of #ref(<thm-expected-pgd-convergence>, supplement: [Theorem]).

#strong[Case 1:] $lr(bar.v.double x_1^(\*) bar.v.double)_2 = 1$.

Without loss of generality, assume (by symmetry) that $(x_1^(\*))_I = 1$ and $(x_1^(\*))_i = 0$ (fix a point on the sphere at the pole) so that $x_1^(\*) = e_I$ (the unit vector! Not the error!). Note that this is not a realistic solution since we already assume the solution comes from a continuous function but we'll use this to illustrate how a warm start from these interpolations can do better than a random start.

$ bb(E)_(g_i tilde.op cal(N)) norm(g - e_I)_2^2 & = bb(E)_(g_i tilde.op cal(N)) [sum_(i = 1)^I (g_i - (e_I)_i)^2]\
 & = bb(E)_(g_i tilde.op cal(N)) [sum_(i = 1)^(I - 1) (g_i - 0)^2 + (g_i - 1)^2]\
 & = sum_(i = 1)^(I - 1) bb(E)_(g_i tilde.op cal(N)) [g_i^2] + bb(E)_(g_i tilde.op cal(N)) [(g_i - 1)^2]\
 & = sum_(i = 1)^(I - 1) 1 + bb(E)_(g_i tilde.op cal(N)) [g_i^2 - 2 g_i + 1]\
 & = I - 1 + (1) - 2 (0) + 1\
 & = I + 1 . $

With some Gaussian concentration, we can square root both sides.

#strong[Case 2:] $lr(bar.v.double x_1^(\*) bar.v.double)_2 = 0$.

Assume the solution is centred so that $x_1^(\*) = 0 in bb(R)^I$. Here, we have the well-known result $ bb(E) [norm(x_1^0 - x_1^(\*))] = bb(E) [norm(g - 0)] = sqrt(I) . $

So either way, our initial error for a scaled or centred problem goes like $tilde.op sqrt(I)$. Since the number of points we have is $I$ is one plus a power of two $I = 2^S + 1$, we #emph[expect] (that is, with high probability) the following convergence $ norm(x_1^K - x_1^(\*)) lt.eq (1 - c)^K sqrt(2^S + 2) . $

So to ensure $norm(x_1^K - x_1^(\*)) lt.eq epsilon.alt$, we need $ (1 - c)^K sqrt(2^S + 2) & lt.eq epsilon.alt\
sqrt(2^S + 2) / epsilon.alt & lt.eq (1 - c)^(- K)\
log (sqrt(2^S + 2) / epsilon.alt) & lt.eq K log ((1 - c)^(- 1))\
frac(log (sqrt(2^S + 2) / epsilon.alt), - log (1 - c)) & lt.eq K\
frac(1 / 2 log (2^S + 2) + log (1 \/ epsilon.alt), - log (1 - c)) & lt.eq K . $

For large $S$, this is approximately $ frac(log (1 \/ epsilon.alt) + S / 2 log (2), - log (1 - c)) lt.eq K . $

=== Expected Resolved Multi-scale Descent Convergence Proof
<sec-expected-resolved-convergence-proof>
Proof of #ref(<thm-expected-resolved-convergence>, supplement: [Theorem]).

First we need to have a handle on how close we are to our coarsest discretization at scale $s = s_0$. The coarsest we could go would be when $s_0 = S$ which would result in $2^(S - s_0 + 1) + 1 = 3$ points total at $t = a \, 1 / 2 (a + b)$, and $b$.

For the centred problem, $x_(s_0)^(\*) = 0 in bb(R)^(2^(S - s_0 + 1) + 1)$, so we would expect a Gaussian initialization to have an initial error of $ bb(E) [norm(x_(s_0)^0 - x_(s_0)^(\*))] = sqrt(2^(S - s_0 + 1) + 1) . $ If we used the scaled problem, it is a bit tricker to consider what happens when we only include $2^(S - s_0 + 1) + 1$ many points, out of the total $2^S + 1$ possible points.

We will actually work out a bound of $sqrt(2^(S - s_0 + 1) + 2)$ by considering the "smoothest" case where $(x_1^(\*))_i = 1 / sqrt(2^S + 1)$ (normalized perfectly to a diagonal), and the "roughest" case where $x_1^(\*) = e_(2^S + 1) in bb(R)^(2^S + 1)$ was aligned to the pole. After discretization, the smooth case still has all the same entries, but there are just less of them ($2^(S - s_0 + 1) + 1$ many). In the rough case, we stay aligned to a pole $x_(s_0)^(\*) = e_(2^(S - s_0 + 1) + 1) in bb(R)^(2^(S - s_0 + 1) + 1)$ since the last entry stays at a one, and the rest are still zero. We use a standard normal initialization of $g in bb(R)^(2^(S - s_0 + 1) + 1)$ in the smaller space. This is identical to choosing the same initialization on the finer grid, and dropping our the coordinates that we skip. So this really is a fair comparison to the regular gradient descent! In the smooth case, $ bb(E) [norm(x_(s_0)^0 - x_(s_0)^(\*))^2] & = sum_(i = 1)^(2^(S - s_0 + 1) + 1) bb(E) (g_i - 1 / sqrt(2^S + 1))^2\
 & = sum_(i = 1)^(2^(S - s_0 + 1) + 1) (bb(E) [g_i^2] - 2 1 / sqrt(2^S + 1) bb(E) [g_i] + frac(1, 2^S + 1) bb(E) [1])\
 & = sum_(i = 1)^(2^(S - s_0 + 1) + 1) (1 - 0 + frac(1, 2^S + 1))\
 & = (2^(S - s_0 + 1) + 1) (1 + frac(1, 2^S + 1))\
 & = (2^(S - s_0 + 1) + 1) (frac(2^S + 2, 2^S + 1)) . $ In the rough case $ bb(E) [norm(x_(s_0)^0 - x_(s_0)^(\*))^2] & = sum_(i = 1)^(2^(S - s_0 + 1) + 1 - 1) bb(E) (g_i - 0)^2 + bb(E) (g_(2^(S - s_0 + 1) + 1) - 1)^2\
 & = 2^(S - s_0 + 1) + 2 . $ For $2^(S - s_0 + 1) gt.eq 3$ (which is the case), $2^(S - s_0 + 1) + 2 > (2^(S - s_0 + 1) + 1) (frac(2^S + 2, 2^S + 1))$ (in fact, it is bigger by $1$ asymptotically as $S$ grows large).

So we are justified in using $sqrt(2^(S - s_0 + 1) + 2)$ as our #emph[expected] bound on $norm(x_(s_0)^0 - x_(s_0)^(\*))$.

This means our descent is $  & norm(x_1^(K_1) - x_1^(\*))\
 & lt.eq sqrt(2^(S - 1)) (1 - c)^(sum_(s = 1)^S K_s) e_S + C sum_(s = 1)^(S - 1) 2^s (1 - c)^(sum_(t = 1)^s K_t)\
 & lt.eq sqrt(2^(S - 1)) (1 - c)^(sum_(s = 1)^S K_s) sqrt(2^(S - S + 1) + 2) + C sum_(s = 1)^(S - 1) 2^s (1 - c)^(sum_(t = 1)^s K_t)\
 & = sqrt(2^(S + 1)) (1 - c)^(sum_(s = 1)^S K_s) + C sum_(s = 1)^(S - 1) 2^s (1 - c)^(sum_(t = 1)^s K_t)\
 & = (1 - c)^(K_1) (sqrt(2^(S + 1)) (1 - c)^(sum_(s = 2)^S K_s) + 2 C + C sum_(s = 2)^(S - 1) 2^s (1 - c)^(sum_(t = 2)^s K_t))\
 & = (1 - c)^(K_1) (sqrt(2^(S + 1)) (1 - c)^(sum_(s = 2)^S K_s) + 2 frac(L abs(a - b), 2 sqrt(2^(S + 1))) + frac(L abs(a - b), 2 sqrt(2^(S + 1))) sum_(s = 2)^(S - 1) 2^s (1 - c)^(sum_(t = 2)^s K_t))\
 & = (1 - c)^(K_1) sqrt(2^(S + 1)) ((1 - c)^(sum_(s = 2)^S K_s) + frac(L abs(a - b), 2 dot.op 2^S) + frac(L abs(a - b), 2 dot.op 2) sum_(s = 2)^(S - 1) (1 - c)^(sum_(t = 2)^s K_t) / 2^(S - s)) . $

=== Cost of Projected Gradient Descent vs Multi-scale Proof
<sec-cost-of-gd-vs-ms-proof>
Proof of #ref(<lem-cost-of-gd-vs-ms>, supplement: [Lemma]).

The total cost of regular descent is given by $ C_(upright("GD")) = C_1 K . $ For multiscale, it is $ C_(upright("MS")) = sum_(s = 1)^S C_s K_s . $ It is reasonable to assume that $ C_s gt.eq 3 / 2 C_(s + 1) $ since one has to compute $2^(S - s + 1) + 1$ entries of $x_s$ at scale $s$. If it is a fixed cost $C$ to compute an entry of $x$, then we have $ C_s = C (2^(S - s + 1) + 1) . $ So $ C_s / C_(s + 1) = frac(2^(S - s + 1) + 1, 2^(S - (s + 1) + 1) + 1) = frac(2^(S - s + 1) + 1, 2^(S - s) + 1) gt.eq 3 / 2 $ since the function $frac(2^(x + 1) + 1, 2^x + 1)$ (for nonnegative $x$) is minimized at $x = 0$, where $x = S - s gt.eq 0$. This gives us the chain, $ C_1 gt.eq 3 / 2 C_2 gt.eq (3 / 2)^2 C_3 gt.eq dots.h gt.eq (3 / 2)^(S - 1) C_S $ or $ C_S lt.eq (2 / 3) C_(S - 1) lt.eq dots.h lt.eq (2 / 3)^(S - 2) C_2 lt.eq (2 / 3)^(S - 1) C_1 . $

This means $ C_(upright("MS")) = sum_(s = 1)^S C_s K_s lt.eq C_1 sum_(s = 1)^S (2 / 3)^(s - 1) K_s . $

=== Re-solved Multi-scale Descent Cost Proof
<sec-resolved-cost-proof>
Proof of #ref(<cor-resolved-cost>, supplement: [Corollary]).

What we need to ensure, for multi-scale to be cheaper, with our mild assumption on cost at each scale, is $ C_(upright("MS")) = sum_(s = 1)^S C_s K_s lt.eq C_1 sum_(s = 1)^S (2 / 3)^(s - 1) K_s lt.eq C_1 K . $

To leave the most budget left for $K_1$, we can make all other $K_s = 1$ which gives $ C_1 sum_(s = 1)^S (2 / 3)^(s - 1) K_s = C_1 (K_1 + sum_(s = 2)^S (2 / 3)^(s - 1)) = C_1 (K_1 + 3 (1 - (2 / 3)^S med)) $

And we can upper bound by considering what happens when we increase the number of scales $S arrow.r oo$ (as in the total number of points $I arrow.r oo$). $ C_(upright("MS")) < C_1 (K_1 + 3) $

So interestingly, we can set $K_1 = K - 3$, and the rest of the scales $K_s = 1$ and still be cheaper!

TODO By hand I've shown that for this setup to give a better accuracy, it does not matter what $K$ is (as long at it is $4$ or bigger). There is just a minimum scale $S$ that is needed. And for well conditioned problems $0.3 lt.eq c lt.eq 1$, a scale bigger than $4$ will do the trick. (Note it is not quite $0.3$, it can be made slightly lower)

=== Expected Freezed Multi-scale Descent Convergence Proof
<sec-expected-freezed-convergence-proof>
Proof of #ref(<thm-expected-freezed-convergence>, supplement: [Theorem]).

For multi-scale, we start out with only $I = 2^(S - s_0 + 1) + 1$ points. Taking the largest scale we can $s_0 = S$, we have an initial error bound of $sqrt(2 + 2) = 2$ giving us the expected error bound

$ norm(e_1^(K_1)) & lt.eq 2 d (K) (1 + d (K))^(S - 1) + C frac(d (K) (sqrt(2^S) (d (K) + 1)^S - sqrt(2) (d (K) + 1)), sqrt(2^S) (d (K) + 1) (sqrt(2) d (K) + sqrt(2) - 1)) $

for $C = L / 2 abs(b - a)$ (see #ref(<thm-expected-pgd-convergence>, supplement: [Theorem])).

If we factor out a $d (K)$, we get $ norm(e_1^(K_1)) & lt.eq d (K) (2 (1 + d (K))^(S - 1) + C frac((sqrt(2^S) (d (K) + 1)^S - sqrt(2) (d (K) + 1)), sqrt(2^S) (d (K) + 1) (sqrt(2) d (K) + sqrt(2) - 1))) . $

=== Freezed Multi-scale Descent Cost Proof
<sec-freezed-cost-proof>
Proof of #ref(<cor-freezed-cost>, supplement: [Corollary]).

Recall our expected regular descent error have the bound $ norm(x_1^K - x_1^(\*)) lt.eq (1 - c)^K sqrt(2^S + 2) . $

From #ref(<thm-expected-freezed-convergence>, supplement: [Theorem]), we have the expected error bound for freezed multi-scale

$ norm(e_1^(K_1)) & lt.eq d (K) (2 (1 + d (K))^(S - 1) + C frac((sqrt(2^S) (d (K) + 1)^S - sqrt(2) (d (K) + 1)), sqrt(2^S) (d (K) + 1) (sqrt(2) d (K) + sqrt(2) - 1))) \, $

so need the thing in the bracket to be less than $sqrt(2^S + 2)$.

In the limit as $K arrow.r oo$, we have that $d (K) arrow.r 0$. So for a very small $d (K)$, we get the bracket expression to be $ 2 + C frac((sqrt(2^S) - sqrt(2)), sqrt(2^S) (sqrt(2) - 1)) . $

After simplifying we get $ 2 + C frac((sqrt(2)^(S - 1) - 1), sqrt(2)^(S - 1) (sqrt(2) - 1)) . $

What $S$ do we need for multi scale to be more accurate? $ 2 + C frac((sqrt(2)^(S - 1) - 1), sqrt(2)^(S - 1) (sqrt(2) - 1)) & < sqrt(2^S + 2)\
C & < (sqrt(2^S + 2) - 2) frac(sqrt(2)^(S - 1) (sqrt(2) - 1), (sqrt(2)^(S - 1) - 1))\
C & < (sqrt(2) - 1) (sqrt(2^S + 2) - 2) sqrt(2)^(S - 1) / (sqrt(2)^(S - 1) - 1) $

The first factor is less than one $sqrt(2) - 1 < 1$, and the last factor is basically $1$ when $S > 6$ so let's use that $ C & < (sqrt(2^S + 2) - 2)\
(C / (sqrt(2) - 1) + 2)^2 - 2 & < 2^S\
log_2 ((C / (sqrt(2) - 1) + 2)^2 - 2) & < S $

note because of our approximation, if the above is #emph[not] satisfied, #emph[then] GD is more accurate. This is not an iff condition.

There are a number of ways to modify it so that it becomes "if this condition holds, then multiscale is better". You could replace $C$ with $C + 0.5$ as a simple one.

As a simple case, if $C = 1 / 2$ (the function is 1-Lipschitz on $[0 \, 1]$), then a scale of $S = 2$ or higher is already better for multi scale.

The above work shows that multi-scale will give a tighter expected final error bound, but we need to make sure multi-scale is cheaper.

Reusing the work shown in #ref(<sec-cost-of-gd-vs-ms-proof>, supplement: [Section]), if we assume each scale has the same number of iterations $K_s = K'$, then we get $ C_(upright("MS")) lt.eq C_1 K' sum_(s = 1)^S (2 / 3)^(s - 1) = C_1 K' 3 (1 - (2 / 3)^S) lt.eq 3 C_1 K' . $

So if we make sure $K' < K / 3$, where $K$ is the number of iterations we perform with regular projected gradient descent, then $ C_(upright("MS")) < C_(G D) . $

The largest integer less than $K / 3$ would be $ceil.l K / 3 ceil.r - 1$.

TODO : Note I think the argument can be tightened to get something like $K' < frac(K, 2 + delta)$ for a small delta, but this idea should be fine.

The way the expected error upper bound works for the freezed multi-scale, the condition on the number of scales needed $S$ is enough to ensure that multi-scale gives a better error bound, assuming we take both methods to large enough iterations $K$.

 
  
#set bibliography(style: "citationstyles/ieee-compressed-in-text-citations.csl") 


#bibliography("references.bib")

