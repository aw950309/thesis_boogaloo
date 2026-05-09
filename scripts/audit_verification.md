# Audit verification — computed vs A2-reported values

Internal cross-check. Not for external distribution.

Source: `notes/notes_paper/T0_audit_outputs/A2_signal_vs_noise_2026-05-05.md`

Tolerances: p-values |diff| < 0.005 · CI bounds < 0.010 · SD < 0.001 · n exact

## Aggregate counts

- uncorrected p<0.05: computed=19  reported=19  **MATCH**
- bonferroni_sig: computed=17  reported=17  **MATCH**
- fdr_sig: computed=19  reported=19  **MATCH**

## Binomial tests

- **all_24**: k=17/24 MATCH | one-sided=0.0320 MATCH (A2: 0.0320) | two-sided=0.0639 MATCH (A2: 0.0639)
- **default**: k=5/8 MATCH | one-sided=0.3633 MATCH (A2: 0.3633) | two-sided=0.7266 MATCH (A2: 0.7266)
- **road**: k=5/8 MATCH | one-sided=0.3633 MATCH (A2: 0.3633) | two-sided=0.7266 MATCH (A2: 0.7266)
- **rail**: k=7/8 MATCH | one-sided=0.0352 MATCH (A2: 0.0352) | two-sided=0.0703 MATCH (A2: 0.0703)
- **default_road**: k=10/16 MATCH | one-sided=0.2272 MATCH (A2: 0.2272) | two-sided=0.4545 MATCH (A2: 0.4545)

## Per-combination row checks


### roe_deer / default / lag
    MATCH    computed=120.000000  reported=120.0000  |diff|=0.000000  n
    MATCH    computed=0.002465  reported=0.0025  |diff|=0.000035  mean_delta
    MATCH    computed=0.006601  reported=0.0066  |diff|=0.000001  sd
    MATCH    computed=0.001272  reported=0.0013  |diff|=0.000028  ci_lo
    MATCH    computed=0.003658  reported=0.0037  |diff|=0.000042  ci_hi
    MATCH    computed=0.000014  reported=0.0000  |diff|=0.000014  wilcoxon_p
    MATCH    computed=0.000331  reported=0.0003  |diff|=0.000031  bonf_p
    MATCH    computed=0.000028  reported=0.0000  |diff|=0.000028  fdr_p

### roe_deer / default / no_lag
    MATCH    computed=120.000000  reported=120.0000  |diff|=0.000000  n
    MATCH    computed=0.011699  reported=0.0117  |diff|=0.000001  mean_delta
    MATCH    computed=0.020256  reported=0.0203  |diff|=0.000044  sd
    MATCH    computed=0.008037  reported=0.0080  |diff|=0.000037  ci_lo
    MATCH    computed=0.015360  reported=0.0154  |diff|=0.000040  ci_hi
    MATCH    computed=0.000000  reported=0.0000  |diff|=0.000000  wilcoxon_p
    MATCH    computed=0.000000  reported=0.0000  |diff|=0.000000  bonf_p
    MATCH    computed=0.000000  reported=0.0000  |diff|=0.000000  fdr_p

### roe_deer / road / lag
    MATCH    computed=120.000000  reported=120.0000  |diff|=0.000000  n
    MATCH    computed=0.003707  reported=0.0037  |diff|=0.000007  mean_delta
    MATCH    computed=0.005088  reported=0.0051  |diff|=0.000012  sd
    MATCH    computed=0.002787  reported=0.0028  |diff|=0.000013  ci_lo
    MATCH    computed=0.004627  reported=0.0046  |diff|=0.000027  ci_hi
    MATCH    computed=0.000000  reported=0.0000  |diff|=0.000000  wilcoxon_p
    MATCH    computed=0.000000  reported=0.0000  |diff|=0.000000  bonf_p
    MATCH    computed=0.000000  reported=0.0000  |diff|=0.000000  fdr_p

### roe_deer / road / no_lag
    MATCH    computed=120.000000  reported=120.0000  |diff|=0.000000  n
    MATCH    computed=0.019415  reported=0.0194  |diff|=0.000015  mean_delta
    MATCH    computed=0.011804  reported=0.0118  |diff|=0.000004  sd
    MATCH    computed=0.017281  reported=0.0173  |diff|=0.000019  ci_lo
    MATCH    computed=0.021548  reported=0.0215  |diff|=0.000048  ci_hi
    MATCH    computed=0.000000  reported=0.0000  |diff|=0.000000  wilcoxon_p
    MATCH    computed=0.000000  reported=0.0000  |diff|=0.000000  bonf_p
    MATCH    computed=0.000000  reported=0.0000  |diff|=0.000000  fdr_p

### roe_deer / rail / lag
    MATCH    computed=120.000000  reported=120.0000  |diff|=0.000000  n
    MATCH    computed=0.051839  reported=0.0518  |diff|=0.000039  mean_delta
    MATCH    computed=0.040711  reported=0.0407  |diff|=0.000011  sd
    MATCH    computed=0.044481  reported=0.0445  |diff|=0.000019  ci_lo
    MATCH    computed=0.059198  reported=0.0592  |diff|=0.000002  ci_hi
    MATCH    computed=0.000000  reported=0.0000  |diff|=0.000000  wilcoxon_p
    MATCH    computed=0.000000  reported=0.0000  |diff|=0.000000  bonf_p
    MATCH    computed=0.000000  reported=0.0000  |diff|=0.000000  fdr_p

### roe_deer / rail / no_lag
    MATCH    computed=120.000000  reported=120.0000  |diff|=0.000000  n
    MATCH    computed=0.081933  reported=0.0819  |diff|=0.000033  mean_delta
    MATCH    computed=0.052623  reported=0.0526  |diff|=0.000023  sd
    MATCH    computed=0.072421  reported=0.0724  |diff|=0.000021  ci_lo
    MATCH    computed=0.091445  reported=0.0914  |diff|=0.000045  ci_hi
    MATCH    computed=0.000000  reported=0.0000  |diff|=0.000000  wilcoxon_p
    MATCH    computed=0.000000  reported=0.0000  |diff|=0.000000  bonf_p
    MATCH    computed=0.000000  reported=0.0000  |diff|=0.000000  fdr_p

### moose / default / lag
    MATCH    computed=120.000000  reported=120.0000  |diff|=0.000000  n
    MATCH    computed=0.007381  reported=0.0074  |diff|=0.000019  mean_delta
    MATCH    computed=0.012738  reported=0.0127  |diff|=0.000038  sd
    MATCH    computed=0.005079  reported=0.0051  |diff|=0.000021  ci_lo
    MATCH    computed=0.009684  reported=0.0097  |diff|=0.000016  ci_hi
    MATCH    computed=0.000000  reported=0.0000  |diff|=0.000000  wilcoxon_p
    MATCH    computed=0.000000  reported=0.0000  |diff|=0.000000  bonf_p
    MATCH    computed=0.000000  reported=0.0000  |diff|=0.000000  fdr_p

### moose / default / no_lag
    MATCH    computed=120.000000  reported=120.0000  |diff|=0.000000  n
    MATCH    computed=0.022437  reported=0.0224  |diff|=0.000037  mean_delta
    MATCH    computed=0.031728  reported=0.0317  |diff|=0.000028  sd
    MATCH    computed=0.016702  reported=0.0167  |diff|=0.000002  ci_lo
    MATCH    computed=0.028172  reported=0.0282  |diff|=0.000028  ci_hi
    MATCH    computed=0.000000  reported=0.0000  |diff|=0.000000  wilcoxon_p
    MATCH    computed=0.000000  reported=0.0000  |diff|=0.000000  bonf_p
    MATCH    computed=0.000000  reported=0.0000  |diff|=0.000000  fdr_p

### moose / road / lag
    MATCH    computed=120.000000  reported=120.0000  |diff|=0.000000  n
    MATCH    computed=0.007453  reported=0.0075  |diff|=0.000047  mean_delta
    MATCH    computed=0.013407  reported=0.0134  |diff|=0.000007  sd
    MATCH    computed=0.005030  reported=0.0050  |diff|=0.000030  ci_lo
    MATCH    computed=0.009877  reported=0.0099  |diff|=0.000023  ci_hi
    MATCH    computed=0.000000  reported=0.0000  |diff|=0.000000  wilcoxon_p
    MATCH    computed=0.000001  reported=0.0000  |diff|=0.000001  bonf_p
    MATCH    computed=0.000000  reported=0.0000  |diff|=0.000000  fdr_p

### moose / road / no_lag
    MATCH    computed=120.000000  reported=120.0000  |diff|=0.000000  n
    MATCH    computed=0.016476  reported=0.0165  |diff|=0.000024  mean_delta
    MATCH    computed=0.035936  reported=0.0359  |diff|=0.000036  sd
    MATCH    computed=0.009980  reported=0.0100  |diff|=0.000020  ci_lo
    MATCH    computed=0.022972  reported=0.0230  |diff|=0.000028  ci_hi
    MATCH    computed=0.000010  reported=0.0000  |diff|=0.000010  wilcoxon_p
    MATCH    computed=0.000248  reported=0.0002  |diff|=0.000048  bonf_p
    MATCH    computed=0.000023  reported=0.0000  |diff|=0.000023  fdr_p

### moose / rail / lag
    MATCH    computed=120.000000  reported=120.0000  |diff|=0.000000  n
    MATCH    computed=0.001993  reported=0.0020  |diff|=0.000007  mean_delta
    MATCH    computed=0.083924  reported=0.0839  |diff|=0.000024  sd
    MATCH    computed=-0.013177  reported=-0.0132  |diff|=0.000023  ci_lo
    MATCH    computed=0.017163  reported=0.0172  |diff|=0.000037  ci_hi
    MATCH    computed=0.221325  reported=0.2213  |diff|=0.000025  wilcoxon_p
    MATCH    computed=1.000000  reported=1.0000  |diff|=0.000000  bonf_p
    MATCH    computed=0.255767  reported=0.2558  |diff|=0.000033  fdr_p

### moose / rail / no_lag
    MATCH    computed=120.000000  reported=120.0000  |diff|=0.000000  n
    MATCH    computed=0.095938  reported=0.0959  |diff|=0.000038  mean_delta
    MATCH    computed=0.133696  reported=0.1337  |diff|=0.000004  sd
    MATCH    computed=0.071771  reported=0.0718  |diff|=0.000029  ci_lo
    MATCH    computed=0.120104  reported=0.1201  |diff|=0.000004  ci_hi
    MATCH    computed=0.000000  reported=0.0000  |diff|=0.000000  wilcoxon_p
    MATCH    computed=0.000000  reported=0.0000  |diff|=0.000000  bonf_p
    MATCH    computed=0.000000  reported=0.0000  |diff|=0.000000  fdr_p

### wild_boar / default / lag
    MATCH    computed=120.000000  reported=120.0000  |diff|=0.000000  n
    MATCH    computed=-0.013948  reported=-0.0139  |diff|=0.000048  mean_delta
    MATCH    computed=0.037140  reported=0.0371  |diff|=0.000040  sd
    MATCH    computed=-0.020662  reported=-0.0207  |diff|=0.000038  ci_lo
    MATCH    computed=-0.007235  reported=-0.0072  |diff|=0.000035  ci_hi
    MATCH    computed=0.000729  reported=0.0007  |diff|=0.000029  wilcoxon_p
    MATCH    computed=0.017502  reported=0.0175  |diff|=0.000002  bonf_p
    MATCH    computed=0.001167  reported=0.0012  |diff|=0.000033  fdr_p

### wild_boar / default / no_lag
    MATCH    computed=120.000000  reported=120.0000  |diff|=0.000000  n
    MATCH    computed=-0.007266  reported=-0.0073  |diff|=0.000034  mean_delta
    MATCH    computed=0.062331  reported=0.0623  |diff|=0.000031  sd
    MATCH    computed=-0.018533  reported=-0.0185  |diff|=0.000033  ci_lo
    MATCH    computed=0.004001  reported=0.0040  |diff|=0.000001  ci_hi
    MATCH    computed=0.805548  reported=0.8055  |diff|=0.000048  wilcoxon_p
    MATCH    computed=1.000000  reported=1.0000  |diff|=0.000000  bonf_p
    MATCH    computed=0.805548  reported=0.8055  |diff|=0.000048  fdr_p

### wild_boar / road / lag
    MATCH    computed=120.000000  reported=120.0000  |diff|=0.000000  n
    MATCH    computed=-0.013621  reported=-0.0136  |diff|=0.000021  mean_delta
    MATCH    computed=0.030109  reported=0.0301  |diff|=0.000009  sd
    MATCH    computed=-0.019063  reported=-0.0191  |diff|=0.000037  ci_lo
    MATCH    computed=-0.008178  reported=-0.0082  |diff|=0.000022  ci_hi
    MATCH    computed=0.000017  reported=0.0000  |diff|=0.000017  wilcoxon_p
    MATCH    computed=0.000405  reported=0.0004  |diff|=0.000005  bonf_p
    MATCH    computed=0.000031  reported=0.0000  |diff|=0.000031  fdr_p

### wild_boar / road / no_lag
    MATCH    computed=120.000000  reported=120.0000  |diff|=0.000000  n
    MATCH    computed=-0.009889  reported=-0.0099  |diff|=0.000011  mean_delta
    MATCH    computed=0.058433  reported=0.0584  |diff|=0.000033  sd
    MATCH    computed=-0.020451  reported=-0.0205  |diff|=0.000049  ci_lo
    MATCH    computed=0.000673  reported=0.0007  |diff|=0.000027  ci_hi
    MATCH    computed=0.234453  reported=0.2345  |diff|=0.000047  wilcoxon_p
    MATCH    computed=1.000000  reported=1.0000  |diff|=0.000000  bonf_p
    MATCH    computed=0.255767  reported=0.2558  |diff|=0.000033  fdr_p

### wild_boar / rail / lag
    MATCH    computed=117.000000  reported=117.0000  |diff|=0.000000  n
    MATCH    computed=0.025862  reported=0.0259  |diff|=0.000038  mean_delta
    MATCH    computed=0.103985  reported=0.1040  |diff|=0.000015  sd
    MATCH    computed=0.006822  reported=0.0068  |diff|=0.000022  ci_lo
    MATCH    computed=0.044903  reported=0.0449  |diff|=0.000003  ci_hi
    MATCH    computed=0.000810  reported=0.0008  |diff|=0.000010  wilcoxon_p
    MATCH    computed=0.019434  reported=0.0194  |diff|=0.000034  bonf_p
    MATCH    computed=0.001215  reported=0.0012  |diff|=0.000015  fdr_p

### wild_boar / rail / no_lag
    MATCH    computed=117.000000  reported=117.0000  |diff|=0.000000  n
    MATCH    computed=0.103541  reported=0.1035  |diff|=0.000041  mean_delta
    MATCH    computed=0.138096  reported=0.1381  |diff|=0.000004  sd
    MATCH    computed=0.078255  reported=0.0783  |diff|=0.000045  ci_lo
    MATCH    computed=0.128828  reported=0.1288  |diff|=0.000028  ci_hi
    MATCH    computed=0.000000  reported=0.0000  |diff|=0.000000  wilcoxon_p
    MATCH    computed=0.000000  reported=0.0000  |diff|=0.000000  bonf_p
    MATCH    computed=0.000000  reported=0.0000  |diff|=0.000000  fdr_p

### fallow_deer / default / lag
    MATCH    computed=119.000000  reported=119.0000  |diff|=0.000000  n
    MATCH    computed=-0.022520  reported=-0.0225  |diff|=0.000020  mean_delta
    MATCH    computed=0.061479  reported=0.0615  |diff|=0.000021  sd
    MATCH    computed=-0.033680  reported=-0.0337  |diff|=0.000020  ci_lo
    MATCH    computed=-0.011360  reported=-0.0114  |diff|=0.000040  ci_hi
    MATCH    computed=0.007753  reported=0.0078  |diff|=0.000047  wilcoxon_p
    MATCH    computed=0.186082  reported=0.1861  |diff|=0.000018  bonf_p
    MATCH    computed=0.010338  reported=0.0103  |diff|=0.000038  fdr_p

### fallow_deer / default / no_lag
    MATCH    computed=119.000000  reported=119.0000  |diff|=0.000000  n
    MATCH    computed=0.015668  reported=0.0157  |diff|=0.000032  mean_delta
    MATCH    computed=0.091356  reported=0.0914  |diff|=0.000044  sd
    MATCH    computed=-0.000916  reported=-0.0009  |diff|=0.000016  ci_lo
    MATCH    computed=0.032251  reported=0.0323  |diff|=0.000049  ci_hi
    MATCH    computed=0.000866  reported=0.0009  |diff|=0.000034  wilcoxon_p
    MATCH    computed=0.020793  reported=0.0208  |diff|=0.000007  bonf_p
    MATCH    computed=0.001223  reported=0.0012  |diff|=0.000023  fdr_p

### fallow_deer / road / lag
    MATCH    computed=119.000000  reported=119.0000  |diff|=0.000000  n
    MATCH    computed=-0.021864  reported=-0.0219  |diff|=0.000036  mean_delta
    MATCH    computed=0.060149  reported=0.0601  |diff|=0.000049  sd
    MATCH    computed=-0.032783  reported=-0.0328  |diff|=0.000017  ci_lo
    MATCH    computed=-0.010945  reported=-0.0109  |diff|=0.000045  ci_hi
    MATCH    computed=0.026451  reported=0.0265  |diff|=0.000049  wilcoxon_p
    MATCH    computed=0.634827  reported=0.6348  |diff|=0.000027  bonf_p
    MATCH    computed=0.033412  reported=0.0334  |diff|=0.000012  fdr_p

### fallow_deer / road / no_lag
    MATCH    computed=119.000000  reported=119.0000  |diff|=0.000000  n
    MATCH    computed=0.023075  reported=0.0231  |diff|=0.000025  mean_delta
    MATCH    computed=0.086267  reported=0.0863  |diff|=0.000033  sd
    MATCH    computed=0.007415  reported=0.0074  |diff|=0.000015  ci_lo
    MATCH    computed=0.038735  reported=0.0387  |diff|=0.000035  ci_hi
    MATCH    computed=0.000212  reported=0.0002  |diff|=0.000012  wilcoxon_p
    MATCH    computed=0.005082  reported=0.0051  |diff|=0.000018  bonf_p
    MATCH    computed=0.000363  reported=0.0004  |diff|=0.000037  fdr_p

### fallow_deer / rail / lag
    MATCH    computed=54.000000  reported=54.0000  |diff|=0.000000  n
    MATCH    computed=-0.067819  reported=-0.0678  |diff|=0.000019  mean_delta
    MATCH    computed=0.164630  reported=0.1646  |diff|=0.000030  sd
    MATCH    computed=-0.112755  reported=-0.1128  |diff|=0.000045  ci_lo
    MATCH    computed=-0.022884  reported=-0.0229  |diff|=0.000016  ci_hi
    MATCH    computed=0.552675  reported=0.5527  |diff|=0.000025  wilcoxon_p
    MATCH    computed=1.000000  reported=1.0000  |diff|=0.000000  bonf_p
    MATCH    computed=0.576705  reported=0.5767  |diff|=0.000005  fdr_p

### fallow_deer / rail / no_lag
    MATCH    computed=54.000000  reported=54.0000  |diff|=0.000000  n
    MATCH    computed=0.009268  reported=0.0093  |diff|=0.000032  mean_delta
    MATCH    computed=0.130672  reported=0.1307  |diff|=0.000028  sd
    MATCH    computed=-0.026399  reported=-0.0264  |diff|=0.000001  ci_lo
    MATCH    computed=0.044934  reported=0.0449  |diff|=0.000034  ci_hi
    MATCH    computed=0.226052  reported=0.2261  |diff|=0.000048  wilcoxon_p
    MATCH    computed=1.000000  reported=1.0000  |diff|=0.000000  bonf_p
    MATCH    computed=0.255767  reported=0.2558  |diff|=0.000033  fdr_p

## Result

Per-row divergences: **0 of 24** combinations had at least one diverging field.
