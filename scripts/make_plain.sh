#!/usr/bin/env bash
#
# Adapted by Shilad Sen from
# https://github.com/facebookresearch/fastText/blob/master/get-wikimedia.sh
#

set -e

if [ -z "$1" ]; then
    echo "usage: $0 language" >&2
    exit 1
fi



normalize_text() {
    sed -e 's/^[ ]*WIKIBRAIN/@WikiBrainDoc/; t' \
        -e "s/’/'/g" -e "s/′/'/g" -e "s/''/ /g" -e "s/'/ ' /g" -e "s/“/\"/g" -e "s/”/\"/g" \
        -e 's/"/ " /g' -e 's/\./ \. /g' -e 's/<br \/>/ /g' -e 's/, / , /g' -e 's/(/ ( /g' -e 's/)/ ) /g' -e 's/\!/ \! /g' \
        -e 's/\?/ \? /g' -e 's/\;/ /g' -e 's/\:/ /g' -e 's/-/ - /g' -e 's/=/ /g' -e 's/=/ /g' -e 's/*/ /g' -e 's/|/ /g' \
        -e 's/«/ /g'  -e 's/[0-9]/ /g'
}


export LANGUAGE=en_US.UTF-8
export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8


wb_lang=$1
s3_dir=$2


ROOT="base_${wb_lang}"
mkdir -p "${ROOT}"
echo "Saving data in ""$ROOT"

wget -c "https://dumps.wikimedia.org/""${wb_lang}""wiki/latest/""${wb_lang}""wiki-latest-pages-articles.xml.bz2" -P "${ROOT}" &&
echo "Processing ""$ROOT"/"${wb_lang}""wiki-latest-pages-articles.xml.bz2" &&
bzip2 -c -d "$ROOT"/"${wb_lang}""wiki-latest-pages-articles.xml.bz2" | perl -e '

sub  trim { my $s = shift; $s =~ s/^\s+|\s+$//g; return $s };

# Program to filter Wikipedia XML dumps to "clean" text consisting only of lowercase
# letters (a-z, converted from A-Z), and spaces (never consecutive)...
# All other characters are converted to spaces.  Only text which normally appears.
# in the web browser is displayed.  Tables are removed.  Image captions are.
# preserved.  Links are converted to normal text.  Digits are spelled out.
# *** Modified to not spell digits or throw away non-ASCII characters ***
# Written by Matt Mahoney, June 10, 2006.  This program is released to the public domain.
$/=">";                     # input record separator
while (<>) {
  if (/<text /i) {$text=1;}  # remove all but between <text> ... </text>
    if (!$article_title && /<\/title/i) {
    s/<.*>//;               # remove xml tags
    $article_title=$_;
  }
  if (!$article_id && /<\/id/i) {
    s/<.*>//;               # remove xml tags
    $article_id=$_;
  }
  if (/#redirect/i) {$text=0;}  # remove #REDIRECT
  if ($text) {
    # Remove any text not normally visible
    if (/<\/text>/i) {$text=0;}
    s/<.*>//;               # remove xml tags
    s/&amp;/&/g;            # decode URL encoded chars
    s/&lt;/</g;
    s/&gt;/>/g;
    s/<ref[^<]*<\/ref>//g;  # remove references <ref...> ... </ref>
    s/<[^>]*>//g;           # remove xhtml tags
    s/\[http:[^] ]*/[/g;    # remove normal url, preserve visible text
    s/\|thumb//ig;          # remove images links, preserve caption
    s/\|left//ig;
    s/\|right//ig;
    s/\|\d+px//ig;
    s/\[\[image:[^\[\]]*\|//ig;
    s/\[\[category:([^|\]]*)[^]]*\]\]/[[$1]]/ig;  # show categories without markup
    s/\[\[[a-z\-]*:[^\]]*\]\]//g;  # remove links to other languages
    s/\[\[[^\|\]]*\|/[[/g;  # remove wiki url, preserve visible text
    s/\{\{[^\}]*\}\}//g;         # remove {{icons}} and {tables}
    s/\{[^}]*\}//g;
    s/\[//g;                # remove [ and ]
    s/\]//g;
    s/&[^;]*;/ /g;          # remove URL encoded chars
    $_=" $_ ";
    chop;
    if (trim($_) && $article_id) {
       print "WIKIBRAIN\t$article_id\t$article_title\n";
       $article_id="";
       $article_title="";
    }
    print $_;
  }
}
' | normalize_text | awk '{if (NF>1) print;}' | tr -s " "  > "${ROOT}"/corpus.txt &&

# Build up dictionary (hack)
sed -E -e 's/[[:blank:]]+/\n/g' "${ROOT}"/corpus.txt |
grep -v '^[ [:punct:]]*$' |
sort |
uniq -c |
sed 's/[ ]*\([0-9][0-9]*\) /w \1 /' |
grep -v '^w [0-4] ' > "${ROOT}"/dictionary.txt &&

pbzip2 -f "${ROOT}"/corpus.txt  &&
pbzip2 -f "${ROOT}"/dictionary.txt  &&
aws s3 cp "${ROOT}"/corpus.txt.bz2 ${s3_dir}/${wb_lang}/  &&
aws s3 cp "${ROOT}"/dictionary.txt.bz2 ${s3_dir}/${wb_lang}/

