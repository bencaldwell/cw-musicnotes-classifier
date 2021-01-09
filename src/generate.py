import os
import sys
import pandas as pd
import argparse
import yaml

table_text = """
<table class="wikitable sortable jquery-tablesorter" style="text-align:center;">
<thead><tr>
<th rowspan="2" width="35px" class="headerSort" tabindex="0" role="columnheader button" title="Sort ascending">Key number
</th>
<th rowspan="2" width="150px" class="headerSort" tabindex="0" role="columnheader button" title="Sort ascending"><a href="/wiki/Helmholtz_pitch_notation" title="Helmholtz pitch notation">Helmholtz</a> name<sup id="cite_ref-:2_5-0" class="reference"><a href="#cite_note-:2-5">[5]</a></sup>
</th>
<th rowspan="2" width="150px" class="headerSort" tabindex="0" role="columnheader button" title="Sort ascending"><a href="/wiki/Scientific_pitch_notation" title="Scientific pitch notation">Scientific</a> name<sup id="cite_ref-:2_5-1" class="reference"><a href="#cite_note-:2-5">[5]</a></sup>
</th>
<th rowspan="2" width="138px" class="headerSort" tabindex="0" role="columnheader button" title="Sort ascending">Frequency (<a href="/wiki/Hertz" title="Hertz">Hz</a>) (Equal temperament) <sup id="cite_ref-6" class="reference"><a href="#cite_note-6">[6]</a></sup>
</th>
<th colspan="5">Corresponding open strings
</th></tr><tr>
<th width="6%" class="headerSort" tabindex="0" role="columnheader button" title="Sort ascending"><a href="/wiki/Violin#Tuning" title="Violin">Violin</a>
</th>
<th width="6%" class="headerSort" tabindex="0" role="columnheader button" title="Sort ascending"><a href="/wiki/Viola#Tuning" title="Viola">Viola</a>
</th>
<th width="7%" class="headerSort" tabindex="0" role="columnheader button" title="Sort ascending"><a href="/wiki/Cello#Strings" title="Cello">Cello</a>
</th>
<th width="9%" class="headerSort" tabindex="0" role="columnheader button" title="Sort ascending"><a href="/wiki/Double_bass#Tuning" title="Double bass">Bass</a>
</th>
<th width="11%" class="headerSort" tabindex="0" role="columnheader button" title="Sort ascending"><a href="/wiki/Guitar_tunings" title="Guitar tunings">Guitar</a>
</th></tr></thead><tbody>

<tr bgcolor="lightgray">
<td>108
</td>
<td>b′′′′′</td>
<td>B<sub>8</sub></td>
<td>7902.133</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>
</td></tr>
<tr>
<td bgcolor="darkslategray" style="color:white">107
</td>
<td bgcolor="lightgray">a<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-sharp">♯</span></span>′′′′′/b<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-flat">♭</span></span>′′′′′
</td>
<td bgcolor="lightgray">A<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-sharp">♯</span></span><sub>8</sub>/B<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-flat">♭</span></span><sub>8</sub>
</td>
<td bgcolor="lightgray">7458.620
</td>
<td bgcolor="lightgray">
</td>
<td bgcolor="lightgray">
</td>
<td bgcolor="lightgray">
</td>
<td bgcolor="lightgray">
</td>
<td bgcolor="lightgray">
</td></tr>
<tr>
<td bgcolor="lightgray">106
</td>
<td bgcolor="lightgray">a′′′′′
</td>
<td bgcolor="lightgray">A<sub>8</sub>
</td>
<td bgcolor="lightgray"><b>7040.000</b>
</td>
<td bgcolor="lightgray">
</td>
<td bgcolor="lightgray">
</td>
<td bgcolor="lightgray">
</td>
<td bgcolor="lightgray">
</td>
<td bgcolor="lightgray">
</td></tr>
<tr>
<td bgcolor="darkslategray" style="color:white">105
</td>
<td bgcolor="lightgray">g<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-sharp">♯</span></span>′′′′′/a<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-flat">♭</span></span>′′′′′
</td>
<td bgcolor="lightgray">G<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-sharp">♯</span></span><sub>8</sub>/A<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-flat">♭</span></span><sub>8</sub>
</td>
<td bgcolor="lightgray">6644.875
</td>
<td bgcolor="lightgray">
</td>
<td bgcolor="lightgray">
</td>
<td bgcolor="lightgray">
</td>
<td bgcolor="lightgray">
</td>
<td bgcolor="lightgray">
</td></tr>
<tr>
<td bgcolor="lightgray">104
</td>
<td bgcolor="lightgray">g′′′′′
</td>
<td bgcolor="lightgray">G<sub>8</sub>
</td>
<td bgcolor="lightgray">6271.927
</td>
<td bgcolor="lightgray">
</td>
<td bgcolor="lightgray">
</td>
<td bgcolor="lightgray">
</td>
<td bgcolor="lightgray">
</td>
<td bgcolor="lightgray">
</td></tr>
<tr>
<td bgcolor="darkslategray" style="color:white">103
</td>
<td bgcolor="lightgray">f<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-sharp">♯</span></span>′′′′′/g<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-flat">♭</span></span>′′′′′
</td>
<td bgcolor="lightgray">F<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-sharp">♯</span></span><sub>8</sub>/G<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-flat">♭</span></span><sub>8</sub>
</td>
<td bgcolor="lightgray">5919.911
</td>
<td bgcolor="lightgray">
</td>
<td bgcolor="lightgray">
</td>
<td bgcolor="lightgray">
</td>
<td bgcolor="lightgray">
</td>
<td bgcolor="lightgray">
</td></tr>
<tr>
<td bgcolor="lightgray">102
</td>
<td bgcolor="lightgray">f′′′′′
</td>
<td bgcolor="lightgray">F<sub>8</sub>
</td>
<td bgcolor="lightgray">5587.652
</td>
<td bgcolor="lightgray">
</td>
<td bgcolor="lightgray">
</td>
<td bgcolor="lightgray">
</td>
<td bgcolor="lightgray">
</td>
<td bgcolor="lightgray">
</td></tr>
<tr>
<td bgcolor="lightgray">101
</td>
<td bgcolor="lightgray">e′′′′′
</td>
<td bgcolor="lightgray">E<sub>8</sub>
</td>
<td bgcolor="lightgray">5274.041
</td>
<td bgcolor="lightgray">
</td>
<td bgcolor="lightgray">
</td>
<td bgcolor="lightgray">
</td>
<td bgcolor="lightgray">
</td>
<td bgcolor="lightgray">
</td></tr>
<tr>
<td bgcolor="darkslategray" style="color:white">100
</td>
<td bgcolor="lightgray">d<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-sharp">♯</span></span>′′′′′/e<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-flat">♭</span></span>′′′′′
</td>
<td bgcolor="lightgray">D<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-sharp">♯</span></span><sub>8</sub>/E<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-flat">♭</span></span><sub>8</sub>
</td>
<td bgcolor="lightgray">4978.032
</td>
<td bgcolor="lightgray">
</td>
<td bgcolor="lightgray">
</td>
<td bgcolor="lightgray">
</td>
<td bgcolor="lightgray">
</td>
<td bgcolor="lightgray">
</td></tr>
<tr>
<td bgcolor="lightgray">99
</td>
<td bgcolor="lightgray">d′′′′′
</td>
<td bgcolor="lightgray">D<sub>8</sub>
</td>
<td bgcolor="lightgray">4698.636
</td>
<td bgcolor="lightgray">
</td>
<td bgcolor="lightgray">
</td>
<td bgcolor="lightgray">
</td>
<td bgcolor="lightgray">
</td>
<td bgcolor="lightgray">
</td></tr>
<tr>
<td bgcolor="darkslategray" style="color:white">98
</td>
<td bgcolor="lightgray">c<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-sharp">♯</span></span>′′′′′/d<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-flat">♭</span></span>′′′′′
</td>
<td bgcolor="lightgray">C<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-sharp">♯</span></span><sub>8</sub>/D<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-flat">♭</span></span><sub>8</sub>
</td>
<td bgcolor="lightgray">4434.922
</td>
<td bgcolor="lightgray">
</td>
<td bgcolor="lightgray">
</td>
<td bgcolor="lightgray">
</td>
<td bgcolor="lightgray">
</td>
<td bgcolor="lightgray">
</td></tr>
<tr>
<td bgcolor="white" style="color:black">88
</td>
<td>c′′′′′ 5-line <a href="/wiki/Octave" title="Octave">octave</a>
</td>
<td><a href="/wiki/Eighth_octave_C" class="mw-redirect" title="Eighth octave C">C<sub>8</sub></a> Eighth octave
</td>
<td>4186.009
</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>
</td></tr>
<tr>
<td bgcolor="white" style="color:black">87
</td>
<td>b′′′′
</td>
<td>B<sub>7</sub>
</td>
<td>3951.066
</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>
</td></tr>
<tr>
<td bgcolor="black" style="color:white">86
</td>
<td>a<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-sharp">♯</span></span>′′′′/b<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-flat">♭</span></span>′′′′
</td>
<td>A<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-sharp">♯</span></span><sub>7</sub>/B<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-flat">♭</span></span><sub>7</sub>
</td>
<td>3729.310
</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>
</td></tr>
<tr>
<td bgcolor="white" style="color:black">85
</td>
<td>a′′′′
</td>
<td>A<sub>7</sub>
</td>
<td><b>3520.000</b>
</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>
</td></tr>
<tr>
<td bgcolor="black" style="color:white">84
</td>
<td>g<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-sharp">♯</span></span>′′′′/a<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-flat">♭</span></span>′′′′
</td>
<td>G<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-sharp">♯</span></span><sub>7</sub>/A<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-flat">♭</span></span><sub>7</sub>
</td>
<td>3322.438
</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>
</td></tr>
<tr>
<td bgcolor="white" style="color:black">83
</td>
<td>g′′′′
</td>
<td>G<sub>7</sub>
</td>
<td>3135.963
</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>
</td></tr>
<tr>
<td bgcolor="black" style="color:white">82
</td>
<td>f<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-sharp">♯</span></span>′′′′/g<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-flat">♭</span></span>′′′′
</td>
<td>F<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-sharp">♯</span></span><sub>7</sub>/G<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-flat">♭</span></span><sub>7</sub>
</td>
<td>2959.955
</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>
</td></tr>
<tr>
<td bgcolor="white" style="color:black">81
</td>
<td>f′′′′
</td>
<td>F<sub>7</sub>
</td>
<td>2793.826
</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>
</td></tr>
<tr>
<td bgcolor="white" style="color:black">80
</td>
<td>e′′′′
</td>
<td>E<sub>7</sub>
</td>
<td>2637.020
</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>
</td></tr>
<tr>
<td bgcolor="black" style="color:white">79
</td>
<td>d<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-sharp">♯</span></span>′′′′/e<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-flat">♭</span></span>′′′′
</td>
<td>D<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-sharp">♯</span></span><sub>7</sub>/E<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-flat">♭</span></span><sub>7</sub>
</td>
<td>2489.016
</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>
</td></tr>
<tr>
<td bgcolor="white" style="color:black">78
</td>
<td>d′′′′
</td>
<td>D<sub>7</sub>
</td>
<td>2349.318
</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>
</td></tr>
<tr>
<td bgcolor="black" style="color:white">77
</td>
<td>c<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-sharp">♯</span></span>′′′′/d<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-flat">♭</span></span>′′′′
</td>
<td>C<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-sharp">♯</span></span><sub>7</sub>/D<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-flat">♭</span></span><sub>7</sub>
</td>
<td>2217.461
</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>
</td></tr>
<tr>
<td bgcolor="white" style="color:black">76
</td>
<td>c′′′′ 4-line <a href="/wiki/Octave" title="Octave">octave</a>
</td>
<td>C<sub>7</sub> <a href="/wiki/Double_high_C" class="mw-redirect" title="Double high C">Double high C</a>
</td>
<td>2093.005
</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>
</td></tr>
<tr>
<td bgcolor="white" style="color:black">75
</td>
<td>b′′′
</td>
<td>B<sub>6</sub>
</td>
<td>1975.533
</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>
</td></tr>
<tr>
<td bgcolor="black" style="color:white">74
</td>
<td>a<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-sharp">♯</span></span>′′′/b<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-flat">♭</span></span>′′′
</td>
<td>A<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-sharp">♯</span></span><sub>6</sub>/B<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-flat">♭</span></span><sub>6</sub>
</td>
<td>1864.655
</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>
</td></tr>
<tr>
<td bgcolor="white" style="color:black">73
</td>
<td>a′′′
</td>
<td>A<sub>6</sub>
</td>
<td><b>1760.000</b>
</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>
</td></tr>
<tr>
<td bgcolor="black" style="color:white">72
</td>
<td>g<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-sharp">♯</span></span>′′′/a<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-flat">♭</span></span>′′′
</td>
<td>G<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-sharp">♯</span></span><sub>6</sub>/A<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-flat">♭</span></span><sub>6</sub>
</td>
<td>1661.219
</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>
</td></tr>
<tr>
<td bgcolor="white" style="color:black">71
</td>
<td>g′′′
</td>
<td>G<sub>6</sub>
</td>
<td>1567.982
</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>
</td></tr>
<tr>
<td bgcolor="black" style="color:white">70
</td>
<td>f<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-sharp">♯</span></span>′′′/g<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-flat">♭</span></span>′′′
</td>
<td>F<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-sharp">♯</span></span><sub>6</sub>/G<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-flat">♭</span></span><sub>6</sub>
</td>
<td>1479.978
</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>
</td></tr>
<tr>
<td bgcolor="white" style="color:black">69
</td>
<td>f′′′
</td>
<td>F<sub>6</sub>
</td>
<td>1396.913
</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>
</td></tr>
<tr>
<td bgcolor="white" style="color:black">68
</td>
<td>e′′′
</td>
<td>E<sub>6</sub>
</td>
<td>1318.510
</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>
</td></tr>
<tr>
<td bgcolor="black" style="color:white">67
</td>
<td>d<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-sharp">♯</span></span>′′′/e<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-flat">♭</span></span>′′′
</td>
<td>D<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-sharp">♯</span></span><sub>6</sub>/E<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-flat">♭</span></span><sub>6</sub>
</td>
<td>1244.508
</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>
</td></tr>
<tr>
<td bgcolor="white" style="color:black">66
</td>
<td>d′′′
</td>
<td>D<sub>6</sub>
</td>
<td>1174.659
</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>
</td></tr>
<tr>
<td bgcolor="black" style="color:white">65
</td>
<td>c<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-sharp">♯</span></span>′′′/d<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-flat">♭</span></span>′′′
</td>
<td>C<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-sharp">♯</span></span><sub>6</sub>/D<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-flat">♭</span></span><sub>6</sub>
</td>
<td>1108.731
</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>
</td></tr>
<tr>
<td bgcolor="white" style="color:black">64
</td>
<td>c′′′ 3-line <a href="/wiki/Octave" title="Octave">octave</a>
</td>
<td>C<sub>6</sub> <a href="/wiki/Soprano_C" class="mw-redirect" title="Soprano C">Soprano C</a> (High C)
</td>
<td>1046.502
</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>
</td></tr>
<tr>
<td bgcolor="white" style="color:black">63
</td>
<td>b′′
</td>
<td>B<sub>5</sub>
</td>
<td>987.7666
</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>
</td></tr>
<tr>
<td bgcolor="black" style="color:white">62
</td>
<td>a<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-sharp">♯</span></span>′′/b<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-flat">♭</span></span>′′
</td>
<td>A<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-sharp">♯</span></span><sub>5</sub>/B<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-flat">♭</span></span><sub>5</sub>
</td>
<td>932.3275
</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>
</td></tr>
<tr>
<td bgcolor="white" style="color:black">61
</td>
<td>a′′
</td>
<td>A<sub>5</sub>
</td>
<td><b>880.0000</b>
</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>
</td></tr>
<tr>
<td bgcolor="black" style="color:white">60
</td>
<td>g<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-sharp">♯</span></span>′′/a<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-flat">♭</span></span>′′
</td>
<td>G<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-sharp">♯</span></span><sub>5</sub>/A<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-flat">♭</span></span><sub>5</sub>
</td>
<td>830.6094
</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>
</td></tr>
<tr>
<td bgcolor="white" style="color:black">59
</td>
<td>g′′
</td>
<td>G<sub>5</sub>
</td>
<td>783.9909
</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>
</td></tr>
<tr>
<td bgcolor="black" style="color:white">58
</td>
<td>f<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-sharp">♯</span></span>′′/g<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-flat">♭</span></span>′′
</td>
<td>F<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-sharp">♯</span></span><sub>5</sub>/G<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-flat">♭</span></span><sub>5</sub>
</td>
<td>739.9888
</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>
</td></tr>
<tr>
<td bgcolor="white" style="color:black">57
</td>
<td>f′′
</td>
<td>F<sub>5</sub>
</td>
<td>698.4565
</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>
</td></tr>
<tr>
<td bgcolor="white" style="color:black">56
</td>
<td>e′′
</td>
<td>E<sub>5</sub>
</td>
<td>659.2551
</td>
<td>E
</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>
</td></tr>
<tr>
<td bgcolor="black" style="color:white">55
</td>
<td>d<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-sharp">♯</span></span>′′/e<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-flat">♭</span></span>′′
</td>
<td>D<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-sharp">♯</span></span><sub>5</sub>/E<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-flat">♭</span></span><sub>5</sub>
</td>
<td>622.2540
</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>
</td></tr>
<tr>
<td bgcolor="white" style="color:black">54
</td>
<td>d′′
</td>
<td>D<sub>5</sub>
</td>
<td>587.3295
</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>
</td></tr>
<tr>
<td bgcolor="black" style="color:white">53
</td>
<td>c<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-sharp">♯</span></span>′′/d<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-flat">♭</span></span>′′
</td>
<td>C<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-sharp">♯</span></span><sub>5</sub>/D<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-flat">♭</span></span><sub>5</sub>
</td>
<td>554.3653
</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>
</td></tr>
<tr>
<td bgcolor="white" style="color:black">52
</td>
<td>c′′ 2-line <a href="/wiki/Octave" title="Octave">octave</a>
</td>
<td>C<sub>5</sub> <a href="/wiki/Tenor_C" class="mw-redirect" title="Tenor C">Tenor C</a>
</td>
<td>523.2511
</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>
</td></tr>
<tr>
<td bgcolor="white" style="color:black">51
</td>
<td>b′
</td>
<td>B<sub>4</sub>
</td>
<td>493.8833
</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>
</td></tr>
<tr>
<td bgcolor="black" style="color:white">50
</td>
<td>a<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-sharp">♯</span></span>′/b<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-flat">♭</span></span>′
</td>
<td>A<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-sharp">♯</span></span><sub>4</sub>/B<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-flat">♭</span></span><sub>4</sub>
</td>
<td>466.1638
</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>
</td></tr>
<tr>
<td bgcolor="white" style="color:black">49
</td>
<td>a′
</td>
<td>A<sub>4</sub> <a href="/wiki/A440_(pitch_standard)" title="A440 (pitch standard)">A440</a>
</td>
<td bgcolor="yellow" style="color:black"><b>440.0000</b>
</td>
<td>A
</td>
<td>A
</td>
<td>
</td>
<td>
</td>
<td>
</td></tr>
<tr>
<td bgcolor="black" style="color:white">48
</td>
<td>g<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-sharp">♯</span></span>′/a<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-flat">♭</span></span>′
</td>
<td>G<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-sharp">♯</span></span><sub>4</sub>/A<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-flat">♭</span></span><sub>4</sub>
</td>
<td>415.3047
</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>
</td></tr>
<tr>
<td bgcolor="white" style="color:black">47
</td>
<td>g′
</td>
<td>G<sub>4</sub>
</td>
<td>391.9954
</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>
</td></tr>
<tr>
<td bgcolor="black" style="color:white">46
</td>
<td>f<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-sharp">♯</span></span>′/g<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-flat">♭</span></span>′
</td>
<td>F<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-sharp">♯</span></span><sub>4</sub>/G<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-flat">♭</span></span><sub>4</sub>
</td>
<td>369.9944
</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>
</td></tr>
<tr>
<td bgcolor="white" style="color:black">45
</td>
<td>f′
</td>
<td>F<sub>4</sub>
</td>
<td>349.2282
</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>
</td></tr>
<tr>
<td bgcolor="white" style="color:black">44
</td>
<td>e′
</td>
<td>E<sub>4</sub>
</td>
<td>329.6276
</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>High E
</td></tr>
<tr>
<td bgcolor="black" style="color:white">43
</td>
<td>d<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-sharp">♯</span></span>′/e<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-flat">♭</span></span>′
</td>
<td>D<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-sharp">♯</span></span><sub>4</sub>/E<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-flat">♭</span></span><sub>4</sub>
</td>
<td>311.1270
</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>
</td></tr>
<tr>
<td bgcolor="white" style="color:black">42
</td>
<td>d′
</td>
<td>D<sub>4</sub>
</td>
<td>293.6648
</td>
<td>D
</td>
<td>D
</td>
<td>
</td>
<td>
</td>
<td>
</td></tr>
<tr>
<td bgcolor="black" style="color:white">41
</td>
<td>c<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-sharp">♯</span></span>′/d<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-flat">♭</span></span>′
</td>
<td>C<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-sharp">♯</span></span><sub>4</sub>/D<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-flat">♭</span></span><sub>4</sub>
</td>
<td>277.1826
</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>
</td></tr>
<tr>
<td bgcolor="white" style="color:black">40
</td>
<td>c′ 1-line <a href="/wiki/Octave" title="Octave">octave</a>
</td>
<td>C<sub>4</sub> <a href="/wiki/C_(musical_note)#Middle_C" title="C (musical note)">Middle C</a>
</td>
<td bgcolor="skyblue" style="color:black">261.6256
</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>
</td></tr>
<tr>
<td bgcolor="white" style="color:black">39
</td>
<td>b
</td>
<td>B<sub>3</sub>
</td>
<td>246.9417
</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>B
</td></tr>
<tr>
<td bgcolor="black" style="color:white">38
</td>
<td>a<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-sharp">♯</span></span>/b<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-flat">♭</span></span>
</td>
<td>A<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-sharp">♯</span></span><sub>3</sub>/B<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-flat">♭</span></span><sub>3</sub>
</td>
<td>233.0819
</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>
</td></tr>
<tr>
<td bgcolor="white" style="color:black">37
</td>
<td>a
</td>
<td>A<sub>3</sub>
</td>
<td><b>220.0000</b>
</td>
<td>
</td>
<td>
</td>
<td>A
</td>
<td>
</td>
<td>
</td></tr>
<tr>
<td bgcolor="black" style="color:white">36
</td>
<td>g<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-sharp">♯</span></span>/a<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-flat">♭</span></span>
</td>
<td>G<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-sharp">♯</span></span><sub>3</sub>/A<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-flat">♭</span></span><sub>3</sub>
</td>
<td>207.6523
</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>
</td></tr>
<tr>
<td bgcolor="white" style="color:black">35
</td>
<td>g
</td>
<td>G<sub>3</sub>
</td>
<td>195.9977
</td>
<td>G
</td>
<td>G
</td>
<td>
</td>
<td>
</td>
<td>G
</td></tr>
<tr>
<td bgcolor="black" style="color:white">34
</td>
<td>f<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-sharp">♯</span></span>/g<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-flat">♭</span></span>
</td>
<td>F<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-sharp">♯</span></span><sub>3</sub>/G<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-flat">♭</span></span><sub>3</sub>
</td>
<td>184.9972
</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>
</td></tr>
<tr>
<td bgcolor="white" style="color:black">33
</td>
<td>f
</td>
<td>F<sub>3</sub>
</td>
<td>174.6141
</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>
</td></tr>
<tr>
<td bgcolor="white" style="color:black">32
</td>
<td>e
</td>
<td>E<sub>3</sub>
</td>
<td>164.8138
</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>
</td></tr>
<tr>
<td bgcolor="black" style="color:white">31
</td>
<td>d<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-sharp">♯</span></span>/e<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-flat">♭</span></span>
</td>
<td>D<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-sharp">♯</span></span><sub>3</sub>/E<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-flat">♭</span></span><sub>3</sub>
</td>
<td>155.5635
</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>
</td></tr>
<tr>
<td bgcolor="white" style="color:black">30
</td>
<td>d
</td>
<td>D<sub>3</sub>
</td>
<td>146.8324
</td>
<td>
</td>
<td>
</td>
<td>D
</td>
<td>
</td>
<td>D
</td></tr>
<tr>
<td bgcolor="black" style="color:white">29
</td>
<td>c<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-sharp">♯</span></span>/d<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-flat">♭</span></span>
</td>
<td>C<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-sharp">♯</span></span><sub>3</sub>/D<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-flat">♭</span></span><sub>3</sub>
</td>
<td>138.5913
</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>
</td></tr>
<tr>
<td bgcolor="white" style="color:black">28
</td>
<td>c small <a href="/wiki/Octave" title="Octave">octave</a>
</td>
<td>C<sub>3</sub>
</td>
<td>130.8128
</td>
<td>
</td>
<td>C
</td>
<td>
</td>
<td>C (6 string)
</td>
<td>
</td></tr>
<tr>
<td bgcolor="white" style="color:black">27
</td>
<td>B
</td>
<td>B<sub>2</sub>
</td>
<td>123.4708
</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>
</td></tr>
<tr>
<td bgcolor="black" style="color:white">26
</td>
<td>A<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-sharp">♯</span></span>/B<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-flat">♭</span></span>
</td>
<td>A<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-sharp">♯</span></span><sub>2</sub>/B<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-flat">♭</span></span><sub>2</sub>
</td>
<td>116.5409
</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>
</td></tr>
<tr>
<td bgcolor="white" style="color:black">25
</td>
<td>A
</td>
<td>A<sub>2</sub>
</td>
<td><b>110.0000</b>
</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>A
</td></tr>
<tr>
<td bgcolor="black" style="color:white">24
</td>
<td>G<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-sharp">♯</span></span>/A<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-flat">♭</span></span>
</td>
<td>G<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-sharp">♯</span></span><sub>2</sub>/A<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-flat">♭</span></span><sub>2</sub>
</td>
<td>103.8262
</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>
</td></tr>
<tr>
<td bgcolor="white" style="color:black">23
</td>
<td>G
</td>
<td>G<sub>2</sub>
</td>
<td>97.99886
</td>
<td>
</td>
<td>
</td>
<td>G
</td>
<td>G
</td>
<td>
</td></tr>
<tr>
<td bgcolor="black" style="color:white">22
</td>
<td>F<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-sharp">♯</span></span>/G<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-flat">♭</span></span>
</td>
<td>F<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-sharp">♯</span></span><sub>2</sub>/G<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-flat">♭</span></span><sub>2</sub>
</td>
<td>92.49861
</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>
</td></tr>
<tr>
<td bgcolor="white" style="color:black">21
</td>
<td>F
</td>
<td>F<sub>2</sub>
</td>
<td>87.30706
</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>
</td></tr>
<tr>
<td bgcolor="white" style="color:black">20
</td>
<td>E
</td>
<td>E<sub>2</sub>
</td>
<td>82.40689
</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>Low E
</td></tr>
<tr>
<td bgcolor="black" style="color:white">19
</td>
<td>D<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-sharp">♯</span></span>/E<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-flat">♭</span></span>
</td>
<td>D<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-sharp">♯</span></span><sub>2</sub>/E<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-flat">♭</span></span><sub>2</sub>
</td>
<td>77.78175
</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>
</td></tr>
<tr>
<td bgcolor="white" style="color:black">18
</td>
<td>D
</td>
<td>D<sub>2</sub>
</td>
<td>73.41619
</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>D
</td>
<td>
</td></tr>
<tr>
<td bgcolor="black" style="color:white">17
</td>
<td>C<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-sharp">♯</span></span>/D<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-flat">♭</span></span>
</td>
<td>C<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-sharp">♯</span></span><sub>2</sub>/D<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-flat">♭</span></span><sub>2</sub>
</td>
<td>69.29566
</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>
</td></tr>
<tr>
<td bgcolor="white" style="color:black">16
</td>
<td>C great <a href="/wiki/Octave" title="Octave">octave</a>
</td>
<td>C<sub>2</sub> <a href="/wiki/Deep_C" class="mw-redirect" title="Deep C">Deep C</a>
</td>
<td>65.40639
</td>
<td>
</td>
<td>
</td>
<td>C
</td>
<td>
</td>
<td>
</td></tr>
<tr>
<td bgcolor="white" style="color:black">15
</td>
<td>B͵
</td>
<td>B<sub>1</sub>
</td>
<td>61.73541
</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>Low B (7 string)
</td></tr>
<tr>
<td bgcolor="black" style="color:white">14
</td>
<td>A<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-sharp">♯</span></span>͵/B<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-flat">♭</span></span>͵
</td>
<td>A<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-sharp">♯</span></span><sub>1</sub>/B<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-flat">♭</span></span><sub>1</sub>
</td>
<td>58.27047
</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>
</td></tr>
<tr>
<td bgcolor="white" style="color:black">13
</td>
<td>A͵
</td>
<td>A<sub>1</sub>
</td>
<td><b>55.00000</b>
</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>A
</td>
<td>
</td></tr>
<tr>
<td bgcolor="black" style="color:white">12
</td>
<td>G<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-sharp">♯</span></span>͵/A<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-flat">♭</span></span>͵
</td>
<td>G<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-sharp">♯</span></span><sub>1</sub>/A<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-flat">♭</span></span><sub>1</sub>
</td>
<td>51.91309
</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>
</td></tr>
<tr>
<td bgcolor="white" style="color:black">11
</td>
<td>G͵
</td>
<td>G<sub>1</sub>
</td>
<td>48.99943
</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>
</td></tr>
<tr>
<td bgcolor="black" style="color:white">10
</td>
<td>F<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-sharp">♯</span></span>͵/G<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-flat">♭</span></span>͵
</td>
<td>F<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-sharp">♯</span></span><sub>1</sub>/G<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-flat">♭</span></span><sub>1</sub>
</td>
<td>46.24930
</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>Low F<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-sharp">♯</span></span> (8 string)
</td></tr>
<tr>
<td bgcolor="white" style="color:black">9
</td>
<td>F͵
</td>
<td>F<sub>1</sub>
</td>
<td>43.65353
</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>
</td></tr>
<tr>
<td bgcolor="white" style="color:black">8
</td>
<td>E͵
</td>
<td>E<sub>1</sub>
</td>
<td>41.20344
</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>E
</td>
<td>
</td></tr>
<tr>
<td bgcolor="black" style="color:white">7
</td>
<td>D<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-sharp">♯</span></span>͵/E<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-flat">♭</span></span>͵
</td>
<td>D<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-sharp">♯</span></span><sub>1</sub>/E<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-flat">♭</span></span><sub>1</sub>
</td>
<td>38.89087
</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>
</td></tr>
<tr>
<td bgcolor="white" style="color:black">6
</td>
<td>D͵
</td>
<td>D<sub>1</sub>
</td>
<td>36.70810
</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>
</td></tr>
<tr>
<td bgcolor="black" style="color:white">5
</td>
<td>C<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-sharp">♯</span></span>͵/D<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-flat">♭</span></span>͵
</td>
<td>C<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-sharp">♯</span></span><sub>1</sub>/D<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-flat">♭</span></span><sub>1</sub>
</td>
<td>34.64783
</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>
</td></tr>
<tr>
<td bgcolor="white" style="color:black">4
</td>
<td>C͵ contra-<a href="/wiki/Octave" title="Octave">octave</a>
</td>
<td>C<sub>1</sub> Pedal C
</td>
<td>32.70320
</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>
</td></tr>
<tr>
<td bgcolor="white" style="color:black">3
</td>
<td>B͵͵
</td>
<td>B<sub>0</sub>
</td>
<td>30.86771
</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>B (5 string)
</td>
<td>
</td></tr>
<tr>
<td bgcolor="black" style="color:white">2
</td>
<td>A<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-sharp">♯</span></span>͵͵/B<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-flat">♭</span></span>͵͵
</td>
<td>A<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-sharp">♯</span></span><sub>0</sub>/B<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-flat">♭</span></span><sub>0</sub>
</td>
<td>29.13524
</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>
</td></tr>
<tr>
<td bgcolor="white" style="color:black">1
</td>
<td>A͵͵
</td>
<td>A<sub>0</sub>
</td>
<td><b>27.50000</b>
</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>
</td>
<td>
</td></tr>
<tr>
<td bgcolor="darkslategray" style="color:white">97
</td>
<td bgcolor="lightgray">G<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-sharp">♯</span></span>͵͵/A<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-flat">♭</span></span>͵͵
</td>
<td bgcolor="lightgray">G<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-sharp">♯</span></span><sub>0</sub>/A<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-flat">♭</span></span><sub>0</sub>
</td>
<td bgcolor="lightgray">25.95654
</td>
<td bgcolor="lightgray">
</td>
<td bgcolor="lightgray">
</td>
<td bgcolor="lightgray">
</td>
<td bgcolor="lightgray">
</td>
<td bgcolor="lightgray">
</td></tr>
<tr>
<td bgcolor="lightgray">96
</td>
<td bgcolor="lightgray">G͵͵
</td>
<td bgcolor="lightgray">G<sub>0</sub>
</td>
<td bgcolor="lightgray">24.49971
</td>
<td bgcolor="lightgray">
</td>
<td bgcolor="lightgray">
</td>
<td bgcolor="lightgray">
</td>
<td bgcolor="lightgray">
</td>
<td bgcolor="lightgray">
</td></tr>
<tr>
<td bgcolor="darkslategray" style="color:white">95
</td>
<td bgcolor="lightgray">F<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-sharp">♯</span></span>͵͵/G<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-flat">♭</span></span>͵͵
</td>
<td bgcolor="lightgray">F<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-sharp">♯</span></span><sub>0</sub>/G<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-flat">♭</span></span><sub>0</sub>
</td>
<td bgcolor="lightgray">23.12465
</td>
<td bgcolor="lightgray">
</td>
<td bgcolor="lightgray">
</td>
<td bgcolor="lightgray">
</td>
<td bgcolor="lightgray">
</td>
<td bgcolor="lightgray">
</td></tr>
<tr>
<td bgcolor="lightgray">94
</td>
<td bgcolor="lightgray">F͵͵
</td>
<td bgcolor="lightgray">F<sub>0</sub>
</td>
<td bgcolor="lightgray">21.82676
</td>
<td bgcolor="lightgray">
</td>
<td bgcolor="lightgray">
</td>
<td bgcolor="lightgray">
</td>
<td bgcolor="lightgray">
</td>
<td bgcolor="lightgray">
</td></tr>
<tr>
<td bgcolor="lightgray">93
</td>
<td bgcolor="lightgray">E͵͵
</td>
<td bgcolor="lightgray">E<sub>0</sub>
</td>
<td bgcolor="lightgray">20.60172
</td>
<td bgcolor="lightgray">
</td>
<td bgcolor="lightgray">
</td>
<td bgcolor="lightgray">
</td>
<td bgcolor="lightgray">
</td>
<td bgcolor="lightgray">
</td></tr>
<tr>
<td bgcolor="darkslategray" style="color:white">92
</td>
<td bgcolor="lightgray">D<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-sharp">♯</span></span>͵͵/E<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-flat">♭</span></span>͵͵
</td>
<td bgcolor="lightgray">D<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-sharp">♯</span></span><sub>0</sub>/E<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-flat">♭</span></span><sub>0</sub>
</td>
<td bgcolor="lightgray">19.44544
</td>
<td bgcolor="lightgray">
</td>
<td bgcolor="lightgray">
</td>
<td bgcolor="lightgray">
</td>
<td bgcolor="lightgray">
</td>
<td bgcolor="lightgray">
</td></tr>
<tr>
<td bgcolor="lightgray">91
</td>
<td bgcolor="lightgray">D͵͵
</td>
<td bgcolor="lightgray">D<sub>0</sub>
</td>
<td bgcolor="lightgray">18.35405
</td>
<td bgcolor="lightgray">
</td>
<td bgcolor="lightgray">
</td>
<td bgcolor="lightgray">
</td>
<td bgcolor="lightgray">
</td>
<td bgcolor="lightgray">
</td></tr>
<tr>
<td bgcolor="darkslategray" style="color:white">90
</td>
<td bgcolor="lightgray">C<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-sharp">♯</span></span>͵͵/D<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-flat">♭</span></span>͵͵
</td>
<td bgcolor="lightgray">C<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-sharp">♯</span></span><sub>0</sub>/D<span class="music-symbol" style="font-family: Arial Unicode MS, Lucida Sans Unicode;"><span class="music-flat">♭</span></span><sub>0</sub>
</td>
<td bgcolor="lightgray">17.32391
</td>
<td bgcolor="lightgray">
</td>
<td bgcolor="lightgray">
</td>
<td bgcolor="lightgray">
</td>
<td bgcolor="lightgray">
</td>
<td bgcolor="lightgray">
</td></tr>
<tr>
<td bgcolor="lightgray">89
</td>
<td bgcolor="lightgray">C͵͵ sub-contra-<a href="/wiki/Octave" title="Octave">octave</a>
</td>
<td bgcolor="lightgray">C<sub>0</sub> Double Pedal C
</td>
<td bgcolor="lightgray">16.35160
</td>
<td bgcolor="lightgray">
</td>
<td bgcolor="lightgray">
</td>
<td bgcolor="lightgray">
</td>
<td bgcolor="lightgray">
</td>
<td bgcolor="lightgray">
</td></tr></tbody><tfoot></tfoot></table>
"""


def load_from_table():
    df = pd.read_html(table_text)[0]
    df.columns = df.columns.get_level_values(1)
    df = df[['Scientific name[5]', 'Frequency (Hz) (Equal temperament) [6]']]
    df.columns = ['name', 'freq']
    return df

if __name__ == "__main__":
    
    params = yaml.safe_load(open('params.yaml'))['generate']

    if len(sys.argv) != 2:
        sys.stderr.write("Arguments error. Usage:\n")
        sys.stderr.write("\tpython generate.py out_file\n")
        sys.exit(1)

    out_file = os.path.abspath(sys.argv[1])
    os.makedirs(out_file, exist_ok=True)

    print(f'out_file: {out_file}')

    