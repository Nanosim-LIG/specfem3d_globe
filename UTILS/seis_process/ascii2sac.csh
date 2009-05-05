#!/bin/csh -f

# Vala Hjorleifsdottir and Qinya Liu, Caltech, Jan 2007

foreach file ($*)
  echo $file
  set nlines = `cat $file | wc -l`
  ./asc2sac $file $nlines $file.sac
end
