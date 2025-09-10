# fixtypst.jl
#
# Script to fix issues when rendering Quarto markdown
# .qmd files to Typst .typ files. These issues are
# because of Pandoc's conversion issues.

filename = "paper"
extention = ".typ"

patterns = (
    # Bad symbol arrangement
    ";." => ".", # citations with sectioning `[@this, Sec. 1] -> @this[Sec. 1];.`
    "];@" => "]@", # citations next to each other with sectioning `@this[Sec. 1];@that`
    " prime" => "'", # plays nicer with variables
    " ; " => " \\; ", # semi-colon in mathmode should be escaped

    # Fix norm and absolute value delimiters
    r"∥([\w\W]+?)∥" => s"norm(\1)",
    r"parallel([\w\W]+?)parallel" => s"norm(\1)",
    r"\\\|([\w\W]+?)\\\|" => s"abs(\1)",
    r"lr\(\|([\w\W]+?)\|\)" => s"abs(\1)",
    r"abs\(\s([\w\W]+?)\s\)" => s"abs(\1)", # remove spaces around absolute value signs

    # Fix inner product and double bracket left-right delimiters
    r"angle.l([\w\W]+?)angle.r" => s"lr(angle.l\1angle.r)",
    r"bracket\.l\.double([\w\W]+?)bracket\.r\.double" => s"lr(bracket.l.double\1bracket.r.double)",
    " | " => " mid(|) ", # midline as part of sets: \left\{x \middle| x > 0 \right\}

    # Figure grids with 4 columns should have a smaller gutter
    "#grid(columns: 4, gutter: 2em" => "#grid(columns: 4, gutter: 1em",

    # Missing callout box icons
    "fa-lightbulb()" => "\"❔\"",
    "fa-info()" => "\"❕\"",
    "fa-exclamation-triangle()" => "\"⚠\"",

    # \dots always gets converted to dots.h, sometimes it should be dots.h.c (TeX's \cdots)
    "times dots.h times" => "times dots.h.c times",
    "times.circle dots.h times.circle" => "times.circle dots.h.c times.circle",

    # Custom citation style in bibliography
    #".bib\")" => ".bib\", style: \"citationstyles/ieee-compressed-in-text-citations.csl\")",

    # Theorem numbering base_level (only show Definition 4.1, not 4.1.1.1 for sub-subsections)
    r"thmbox\(([\w\W]+?)\)" => s"thmbox(\1, base_level: 1)"
    )

"""
    fixit(filename, patterns; extention=".typ", append_filename="_fixed")

Uses patterns to perform find-and-replace on `filename`.


Filename should be given without the file extension. Pattern can be any iterable of Pairs
from a string to a string. The strings can be plain `String` types, or a pair of `Regex`
and `SubstitutionString` using `r"a regex string"` and `s"substitution string"`.

Example
-------
`patterns = (r"∥([\\w\\W]+?)∥" => s"norm(\\1)",)`
"""
function fixit(filename, patterns;
    extension=".typ",
        append_filename=""#"_fixed"
        )
    f = read(filename*extension, String)
    f_new = f
    for pattern in patterns
        f_new = replace(f_new, pattern)
    end
    write(filename*append_filename*extension, f_new)
end

fixit(filename, patterns)

@info "$filename was successfully fixed!"
