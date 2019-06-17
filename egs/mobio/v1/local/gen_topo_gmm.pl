#!/usr/bin/env perl

# Copyright 2012  Johns Hopkins University (author: Daniel Povey)

# Generate a topology file.  This allows control of the number of states in the
# non-silence HMMs, and in the silence HMMs.

if (@ARGV != 6) {
  print STDERR "Usage: utils/gen_topo.pl <num-word-states> <num-freetext-states> <num-silence-states> <colon-separated-word-phones> <colon-separated-freetext-phones> <colon-separated-silence-phones>\n";
  print STDERR "e.g.:  utils/gen_topo.pl 3 5 4:5:6:7:8:9:10 1:2:3\n";
  exit (1);
}

($num_word_states, $num_freetext_states, $num_sil_states, $word_phones, $freetext_phones, $sil_phones) = @ARGV;

( $num_word_states >= 1 && $num_word_states <= 100 ) ||
  die "Unexpected number of word-model states $num_word_states\n";
( $num_freetext_states >= 1 && $num_freetext_states <= 100 ) ||
  die "Unexpected number of freetext-model states $num_freetext_states\n";
(( $num_sil_states == 1 || $num_sil_states >= 3) && $num_sil_states <= 100 ) ||
  die "Unexpected number of silence-model states $num_sil_states\n";

$word_phones =~ s/:/ /g;
$freetext_phones =~ s/:/ /g;
$sil_phones =~ s/:/ /g;
$word_phones =~ m/^\d[ \d]*$/ || die "$0: bad arguments @ARGV\n";
$freetext_phones =~ m/^\d[ \d]*$/ || die "$0: bad arguments @ARGV\n";
$sil_phones =~ m/^\d[ \d]*$/ || die "$0: bad arguments @ARGV\n";

print "<Topology>\n";
print "<TopologyEntry>\n";
print "<ForPhones>\n";
print "$word_phones\n";
print "</ForPhones>\n";
for ($state = 0; $state < $num_word_states; $state++) {
  $statep1 = $state+1;
  print "<State> $state <PdfClass> $state <Transition> $state 0.75 <Transition> $statep1 0.25 </State>\n";
}
print "<State> $num_word_states </State>\n"; # non-emitting final state.
print "</TopologyEntry>\n";

print "<TopologyEntry>\n";
print "<ForPhones>\n";
print "$freetext_phones\n";
print "</ForPhones>\n";
for ($state = 0; $state < $num_freetext_states; $state++) {
  $statep1 = $state+1;
  print "<State> $state <PdfClass> $state <Transition> $state 0.75 <Transition> $statep1 0.25 </State>\n";
}
print "<State> $num_freetext_states </State>\n"; # non-emitting final state.
print "</TopologyEntry>\n";

# Now silence phones.  They have a different topology-- apart from the first and
# last states, it's fully connected, as long as you have >= 3 states.

if ($num_sil_states > 1) {
  $transp = 1.0 / ($num_sil_states-1);
  print "<TopologyEntry>\n";
  print "<ForPhones>\n";
  print "$sil_phones\n";
  print "</ForPhones>\n";
  print "<State> 0 <PdfClass> 0 ";
  for ($nextstate = 0; $nextstate < $num_sil_states-1; $nextstate++) { # Transitions to all but last
    # emitting state.
    print "<Transition> $nextstate $transp ";
  }
  print "</State>\n";
  for ($state = 1; $state < $num_sil_states-1; $state++) { # the central states all have transitions to
    # themselves and to the last emitting state.
    print "<State> $state <PdfClass> $state ";
    for ($nextstate = 1; $nextstate < $num_sil_states; $nextstate++) {
      print "<Transition> $nextstate $transp ";
    }
    print "</State>\n";
  }
  # Final emitting state (non-skippable).
  $state = $num_sil_states-1;
  print "<State> $state <PdfClass> $state <Transition> $state 0.75 <Transition> $num_sil_states 0.25 </State>\n";
  # Final nonemitting state:
  print "<State> $num_sil_states </State>\n";
  print "</TopologyEntry>\n";
} else {
  print "<TopologyEntry>\n";
  print "<ForPhones>\n";
  print "$sil_phones\n";
  print "</ForPhones>\n";
  print "<State> 0 <PdfClass> 0 ";
  print "<Transition> 0 0.75 ";
  print "<Transition> 1 0.25 ";
  print "</State>\n";
  print "<State> $num_sil_states </State>\n"; # non-emitting final state.
  print "</TopologyEntry>\n";
}

print "</Topology>\n";
