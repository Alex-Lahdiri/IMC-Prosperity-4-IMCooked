"""
=============================================================================
 ROUND 2 ALGORITHM — v5 (combined + data-driven improvements)

 OSMIUM strategy:
   Passive MM at fair=10000, MM_HALF=5 (R2 backtester sweep winner).
   Tight ladder (offset=2, split=0.8) concentrates budget at primary quote.
   Nonlinear inventory skew + EOD flush from NoDipAlgo carried forward.

 PEPPER strategy:
   Bullish drift-rider, slope = 1/1000 per tick (confirmed exactly on all
   3 wiki data days; residual std ≈ 2.5 ticks — essentially no crashes).
   BUY_EDGE raised 8→9: wiki data shows NPC asks drift from mid+6.5 on
   day -1 to mid+7.5 on day 1. BUY_EDGE=9 captures 95% of fills on day 1
   vs only 72% at BUY_EDGE=8.
   Regime detection (BULLISH/HOLD/BEARISH) guards against unseen scenarios.
   Scratch-sell at max position locks in realised PnL without abandoning drift.
   MLP gated to BULLISH + high confidence only (data leakage risk noted).

 Market Access Fee: bid() returns 8,000 XIRECs.
   Bid is in XIRECs units (same as PnL). Extra flow value ~20-25k XIRECs;
   8k is EV-positive as long as it clears the median bid.
=============================================================================
"""

import json
import math
from typing import Any, List, Dict
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState


# =============================================================================
#  MLP MODEL  (weights unchanged — leakage risk noted, gated to BULLISH only)
# =============================================================================

def MLP_relu(vec):
    return [x if x > 0.0 else 0.0 for x in vec]

def MLP_dot(a, b):
    s = 0.0
    for x, y in zip(a, b):
        s += x * y
    return s

def MLP_linear(x, W, b):
    return [MLP_dot(row, x) + bias for row, bias in zip(W, b)]

def MLP_scale(x, mean, scale):
    return [(xi - m) / (s if abs(s) > 1e-8 else 1.0) for xi, m, s in zip(x, mean, scale)]

def MLP_forward_logits(x, mean, scale, W1, B1, W2, B2, W3, B3, W4, B4):
    x = MLP_scale(x, mean, scale)
    h1 = MLP_relu(MLP_linear(x, W1, B1))
    h2 = MLP_relu(MLP_linear(h1, W2, B2))
    h3 = MLP_relu(MLP_linear(h2, W3, B3))
    return MLP_linear(h3, W4, B4)

def softmax(logits):
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    s = sum(exps)
    return [e / s for e in exps]

def argmax_index(xs):
    best_i, best_v = 0, xs[0]
    for i in range(1, len(xs)):
        if xs[i] > best_v:
            best_v = xs[i]
            best_i = i
    return best_i

ROOT_MEAN = [11046.184, 11.528486, 11058.738, 11.513417, 0.035091341, 0.18154608, 11052.446, 12.554856, 0.015068635, 0.00081967086, 0.00094773632, 0.015529652, 25.158634, 25.208123, -0.0010554138, 0.10893797, 12.554856, 0.00088695083]
ROOT_SCALE = [606.61921, 4.1845198, 607.23678, 4.1988762, 0.18457074, 0.99558632, 606.91519, 2.4591637, 5.66533, 0.19254344, 1.3256817, 1.0256498, 8.3658242, 8.3470114, 0.26966272, 1.6898706, 2.4591427, 0.19255779]
ROOT_CLASS_VALUES = [-3.0, -1.5, -0.5, 0.0, 0.5, 1.5, 3.0]
ROOT_W1 = [[0.097483993, -0.24760309, 0.28433034, 0.1443373, 0.0069319508, -0.08148133, 0.25124788, 0.40450183, -0.15627442, -0.12604317, -0.0079672681, 0.2761561, 0.18726562, 0.071439579, 0.14608286, 0.3108249, 0.018533345, -0.1434996], [-0.10890052, 0.11745648, 0.082734592, -0.074323528, -0.35588232, -0.26229894, 0.19296007, -0.12651499, -0.11384115, 0.028179564, -0.24275807, 0.012562717, 0.024468726, -0.034194272, -0.10287659, -0.36567271, 0.22024594, 0.090016045], [0.12116425, -0.23179305, -0.25735292, -0.28309765, 0.035035379, 0.025001612, 0.087465599, 0.25425938, -0.050682828, 0.25920382, -0.60411638, -0.0075848629, -0.04287063, 0.079557993, 0.098551936, 0.096547626, -0.049719185, -0.037834764], [-0.00085535593, -0.06249224, 0.028187782, -0.059176054, 0.12522821, -0.039030403, 0.13238493, 0.0042682099, -0.23556414, -0.04940727, -0.24764042, -0.066110708, -0.24169299, -0.15369505, 0.0054080263, 0.49312618, 0.2172412, -0.24411543], [-0.073130988, 0.3116172, 0.16282548, 0.098977774, -0.030998267, 0.032828879, 0.26158407, -0.61340624, -0.15140222, 0.044269811, -0.18513459, 0.020470073, -0.21795125, -0.027484141, 0.10170525, 0.016696004, -0.013970361, -0.0071698227], [-0.14990564, -0.33535224, -0.15022421, 0.11482893, -0.19777635, 0.030847695, 0.25944558, 0.18034907, -0.0589188, -0.051655151, -0.35647127, 0.40761915, -0.054459177, 0.048221257, -0.11770691, -0.025089094, -0.081673242, -0.17420243], [0.079597823, 0.12680343, 0.076929204, -0.26147285, -0.0053174682, -0.1112105, -0.043002382, 0.2271795, 0.080574684, 0.028655875, 0.46492094, 0.19332968, -0.22429833, 0.10131545, 0.18673629, 0.040745061, -0.26271802, 0.20113848], [-0.14262833, -0.11032946, -0.23838675, -0.11863355, -0.047758937, -0.055154014, 0.036880631, -0.25159889, 0.18143933, -0.26117092, 0.34545988, 0.16823986, 0.13762429, -0.1195778, -0.16176191, 0.55938798, 0.33715248, -0.2230709], [0.23873776, 0.0087226527, -0.20546213, -0.20274317, -0.39491045, -0.25108832, 0.19056061, 0.28069562, 0.042146008, -0.049255259, -0.34317008, -0.18312442, 0.28899786, 0.18109104, 0.23528662, 0.017115433, -0.029615456, -0.032524984], [0.010954436, -0.098928668, -0.063434005, -0.22373573, -0.23056251, 0.13380861, -0.012820852, -0.035027597, -0.30027604, 0.062997594, -0.048190244, 0.11948332, -0.11789166, 0.11878232, 0.1375235, -0.35377762, 0.24840909, 0.10854336], [0.21609809, -0.35600474, -0.077500559, 0.27568701, -0.037084889, -0.051954623, -0.077957548, -0.54905802, -0.15762991, -0.37734205, 0.16409174, -0.071164481, 0.036202852, 0.074245289, 0.0065747001, 0.2038199, 0.090368599, -0.14936532], [0.15081507, 0.35991126, 0.043656614, 0.098669961, 0.19666457, -0.076294847, -0.0081192218, 0.077992424, -0.10891651, -0.025955543, -0.42429987, 0.20508784, -0.010680667, -0.17746854, 0.16046101, -0.12256797, -0.2743836, -0.1999798], [-0.20652516, -0.13192581, 0.051438652, 0.32352734, 0.034749832, 0.042410791, 0.045143083, -0.064799227, -0.32300684, -0.50438803, -0.45299026, 0.0062010195, -0.0076353145, -0.044883206, -0.014997377, 0.32904711, 0.14333709, -0.19577795], [0.12103708, -0.25204033, -0.060328081, 0.21273795, -0.0020790347, -0.08258941, 0.17236757, -0.47214049, -0.324218, -0.23590235, -0.25554785, 0.0114112, 0.11253262, -0.16247931, -0.18624695, 0.39220226, 0.16089459, -0.2374071], [0.061725836, 0.13929641, -0.10632228, 0.022771139, -0.074062817, 0.024071503, 0.17148197, -0.662916, 0.019471511, 0.22404423, -0.32289293, 0.00092562672, 0.22098577, -0.062528566, -0.17886105, -0.087256722, 0.028643612, 0.14583538], [-0.0083750486, 0.16636276, 0.14554094, -0.21024068, 0.013830924, 0.0016302193, 0.067004338, 0.33662531, 0.20311396, 0.269853, 0.30528218, 0.0131214, -0.071447261, 0.0059136669, 0.067685083, -0.1710491, -0.049332473, 0.13416542], [-0.0063390611, -0.16698042, -0.11020391, 0.3031652, 0.045350909, -2.3066323e-06, -0.022339996, -0.023140498, -0.36208224, -0.5076198, -0.13394973, 0.026231848, 0.1461439, -0.22413757, -0.23004596, 0.18583, 0.057628766, -0.12178035], [-0.063626446, 0.28637886, -0.14483735, -0.15641825, -0.00087668578, -0.018411431, 0.17691353, -0.20830701, 0.13771053, 0.43508056, 0.15724489, 0.017924521, -0.062478606, 0.038143717, 0.08489684, -0.32746404, 0.067934774, 0.21406075], [0.13477701, 0.018606121, 0.064402521, -0.015869536, -0.11039691, -0.044867165, -0.12293158, -0.26961404, 0.072075643, -0.018464159, 0.12127201, 0.13111356, -0.019819124, -0.04080997, 0.087081648, 0.53026563, 0.072058894, -0.26063299], [0.2763291, -0.12503479, 0.11746147, 0.15391904, -0.35338885, 0.3362267, -0.10500616, 0.14721973, 0.15211251, 0.084021784, -0.34059572, -0.13853917, 0.065419585, 0.43596941, -0.22950868, -0.14056996, 0.053529952, -0.044829708], [0.15939038, -0.039463807, -0.12104439, 0.19972271, -0.34551972, -0.31402117, -0.26978302, -0.11411764, -0.069811046, 0.18054381, -0.095880114, -0.43182236, -0.044162296, -0.10175212, 0.071346372, 0.49337611, 0.31965351, -0.056105252], [-0.11207391, 0.080520481, 0.11591686, -0.058452308, -0.14085047, 0.14080217, -0.064821079, -0.15283579, 0.087720357, -0.10729352, 0.21512927, 0.054239657, -0.086811349, 0.024425749, 0.043541927, -0.38965508, 0.19044651, 0.1234746], [0.12614003, 0.31195024, -0.075537652, -0.20743424, 0.001449783, -0.053339627, 0.041502669, -0.24045548, 0.15484518, 0.41094726, 0.2068951, 0.044999756, -0.15070748, 0.22998808, 0.25842744, -0.087826729, 0.019729951, 0.024864184], [0.096650951, -0.053534221, -0.012349042, 0.15828359, 0.095052734, 0.041702926, -0.13182293, -0.47589481, -0.0039647045, 0.35728368, -0.28598019, -0.027350048, -0.27246895, -0.17060207, 0.22321379, -0.51534408, 0.24930236, 0.31151536], [0.20821527, -0.23418613, -0.084875792, -0.25441805, 0.11405019, -0.054340746, 0.1611973, -0.058334075, 0.077663682, 0.30184749, 0.012700199, 0.1015569, -0.15587661, -0.37939695, 0.00032742252, -0.13704616, -0.059679333, 0.2433047], [-0.10764647, -0.088129871, 0.21256757, -0.11699259, -0.071755275, 0.19801241, -0.12372266, -0.2530672, 0.016467353, 0.19401914, -0.11445415, -0.0059535713, -0.23564759, 0.16102436, -0.06713561, 0.098875992, 0.0027606557, -0.23482902], [-0.1690678, -0.060625888, -0.067128576, 0.25585309, -0.28721693, -0.17911026, -0.21132101, 0.49104378, 0.003107606, -0.015216294, -0.1176535, -0.46296656, -0.031882472, -0.029444436, -0.052806359, 0.057126407, -0.071833871, -0.064974725], [-0.25128117, 0.0071800668, -0.28406408, 0.033248618, -0.031347807, 0.058926743, -0.21857856, 0.11272127, 0.13428171, -0.064932391, 0.027987041, -0.13811031, 0.44806013, -0.068587631, 0.23274164, 0.14658897, 0.0080284644, -0.10455304], [-0.01606198, -0.04351671, -0.062235348, -0.17113023, 0.11345445, 0.016212773, 0.040967803, -0.15446751, 0.11556736, 0.031388577, -0.057866052, 0.011524069, -0.21439816, 0.063500069, -0.17784967, -0.1782757, -0.047578175, -0.036109366], [0.18689924, 0.11485105, -0.09570051, -0.1647502, -0.074696019, 0.10074486, -0.022197628, -0.15568055, -0.13171913, 0.18269131, 0.11811297, 0.0099627674, 0.11015043, -0.1316333, -0.14580277, -0.4079124, 0.25978187, 0.053014807], [-0.005388902, 0.16359992, 0.055397939, 0.22800407, -0.17757857, 0.13883579, 0.14556162, -0.65194303, 0.014081775, -0.20131958, -0.049878456, 0.0021319822, -0.047538213, -0.11487292, -0.045512415, -0.039798286, -0.0023345849, 0.03145127], [-0.13218056, -0.30160275, -0.10356751, -0.074369989, -0.29300663, -0.21656026, 0.16706103, 0.17265919, 0.01661888, 0.17243864, -0.57363623, -0.44085038, -0.16469239, -0.060524508, 0.01462147, 0.094717413, -0.082371749, -0.1676911]]
ROOT_B1 = [-0.07897836, 0.32787889, 0.42205203, 0.02509266, 0.37171128, 0.24426447, 0.16477662, 0.24924842, 0.17361364, 0.41782272, -0.19971856, 0.043153591, -0.34466934, -0.34990299, 0.039343916, -0.076905958, -0.23475201, -0.2170856, 0.34215438, 0.32402053, 0.23767987, -0.1237615, -0.10260084, -0.35970312, 0.21416493, 0.1953063, 0.46144876, 0.30662328, 0.189115, 0.13632874, -0.41334867, 0.19403063]
ROOT_W2 = [[0.056133535, -0.18644017, 0.26839775, 0.2140606, -0.4672327, 0.23567779, -0.31072482, -0.42562917, -0.098972976, -0.23490363, -0.22194809, 0.28696871, 0.17899521, -0.14928038, -0.52684367, -0.50353825, 0.13498834, -0.078251809, -0.2261916, -0.086738691, 0.16715556, -0.33102301, -0.15329857, -0.49850261, -0.39831561, -0.32418016, 0.28593701, 0.10607886, -0.25998089, -0.54412395, -0.82706618, 0.26962051], [0.098333947, 0.12619808, 0.068116382, -0.064766027, 0.11593044, -0.06042023, 0.073481716, -0.048385348, 0.092176847, -0.061094075, -0.26921412, -0.016920708, -0.21545036, -0.21116708, -0.008688001, -0.073463395, -0.34795961, -0.027059894, 0.20936258, 0.27683759, -0.11302032, 0.17696945, 0.087085113, 0.40052626, 0.15899645, 0.036716767, 0.018945217, -0.071091317, 0.061020724, 0.23409991, 0.039860636, -0.055864204], [0.19122532, -0.018733002, -0.045725022, 0.14785819, 0.40029484, 0.018554384, -0.2442846, -0.076754853, -0.0073575233, -0.18717611, 0.070881911, 0.10726088, 0.019279122, 0.12528282, 0.38276055, -0.12586768, 0.020453773, 0.14833425, 0.32643357, -0.042069867, 0.069451623, 0.091027446, 0.073125184, 0.37123698, -0.036158632, 0.11106638, -0.073987253, -0.18697728, 0.066000909, 0.024166128, 0.53836399, 0.0059274794], [-0.57436681, 0.206095, -0.34241101, -0.46386477, 0.25107932, -0.46984801, -0.052789092, -0.40238014, -0.23004507, 0.19915675, 0.037143782, -0.28807595, -0.26115835, 0.091652617, 0.25917572, 0.076591842, 0.0830786, 0.0017549631, -0.46356878, 0.044127591, -0.4300344, 0.25122711, -0.015813647, 0.42080712, 0.21525601, -0.031192731, 0.079650603, -0.11569627, 0.14950249, -0.062947698, 0.37360451, -0.52458131], [0.25868359, 0.09975931, -0.015897688, -0.12290129, 0.24466789, -0.060279582, -0.13089731, -0.3049418, 0.099764645, 0.022699574, -0.23232099, -0.16484644, -0.042053543, -0.25333482, 0.18226556, 0.028924668, -0.28130832, 0.15034004, -0.13440557, 0.048225008, -0.059840143, 0.14398465, 0.0047188797, 0.23137149, -0.10035424, 0.036276888, 0.010576699, -0.059295185, 0.065091595, 0.19290666, 0.16517386, -0.071998775], [0.17785472, 0.12847808, 0.052162111, -0.025096493, 0.078072779, -0.12537985, 0.088666856, 0.037009187, 0.042227857, 0.17113201, -0.13947405, -0.0043268702, -0.35891423, -0.36533841, 0.06139544, -0.014982713, -0.37114531, 0.030689212, 0.11071387, 0.10405456, -0.05921939, 0.11903078, -0.09630961, 0.24397221, -0.040748142, 0.15149313, 0.040194072, 0.07801789, 0.16985291, -0.056672022, -0.21505174, -0.032849714], [0.038748607, -0.27667722, -0.090313196, 0.20697257, 0.072560534, 0.11841165, -0.40251064, 0.50629497, -0.14608428, -0.29362568, 0.10706278, 0.00094344682, 0.084703341, 0.22361501, 0.21575429, -0.36384732, 0.13434978, -0.31164107, 0.32887873, -0.33618423, 0.18592259, -0.54063946, -0.066870965, -0.28484756, -0.48684165, 0.17777941, 0.20851649, 0.25713658, 0.081229277, -0.39176983, 0.39113247, -0.037498765], [-0.055164736, 0.0071678227, 0.19206518, -0.060705941, 0.11451526, 0.080740243, 0.081186786, 0.25339794, 0.14144534, 0.043579657, -0.33661166, -0.17956837, -0.36410251, -0.3013308, -0.094860114, -0.10806549, -0.52859783, -0.42033035, 0.27996868, 0.15296036, 0.035741821, -0.34528109, -0.46777931, 0.22260238, 0.021192512, 0.029289, -0.10630799, 0.011891492, 0.053873911, -0.074805729, -0.39686522, 0.070543423], [0.055328537, -0.52716553, -0.47691438, 0.15048774, 0.36744887, -0.20763738, -0.35320538, 0.25531787, -0.44667739, -0.44449732, 0.25365835, -0.058474105, 0.11317138, 0.19521856, 0.16605505, -0.2243479, 0.064163581, 0.11280508, 0.37042788, -0.53137237, 0.11338305, -0.3520484, 0.25920391, -0.092235617, -0.17723878, 0.28840446, -0.17885178, 0.079995029, 0.19634399, -0.28542695, 0.42897320, -0.20271568], [-0.52441788, 0.17049727, -0.59236348, -0.51032281, 0.077080719, -0.43312708, 0.14796241, -0.50039905, -0.3753413, 0.24151741, 0.085118994, -0.073337689, -0.075291775, -0.002287366, 0.33361378, 0.027926689, -0.029726356, 0.0097298278, -0.44613546, 0.043728009, -0.30342117, 0.032563828, 0.013120897, 0.21973364, 0.074231289, -0.063031211, 0.32786345, -0.11402845, 0.069990315, 0.12291405, 0.33279425, -0.21176104], [0.016598199, 0.068915449, 0.23942213, -0.054496635, 0.021434598, 0.039491102, -0.23797551, 0.046408612, 0.16833818, 0.13699856, -0.18646383, 0.24864072, -0.043042723, -0.14089236, 0.10542005, -0.30904168, 0.020977821, -0.31157553, 0.0044035227, 0.094565816, 0.18126598, -0.23987633, -0.27985504, -0.043053899, 0.032347324, 0.020432001, 0.25098923, 0.22074918, 0.044933148, -0.1810874, -0.23382543, 0.27942368], [0.19721515, 0.14129065, 0.15120956, 0.21857429, 0.14339779, 0.080940001, -0.13840018, 0.011198866, 0.28673369, -0.046509068, -0.16790296, 0.12952729, -0.01162915, -0.071370862, -0.094107084, -0.19851656, -0.05479417, -0.36215174, 0.21879232, 0.23846802, 0.044720747, -0.30970818, -0.46877414, -0.041409435, 0.093182325, -0.10172518, -0.1092156, -0.19133931, -0.039029298, -0.23456353, -0.38952959, 0.21151198], [0.1177135, 0.18068612, -0.049019039, -0.14700629, 0.24516498, -0.00046435778, 0.0997274, -0.13701655, 0.1692265, 0.20956497, -0.35658905, -0.057022125, -0.37405667, -0.37686509, 0.11335208, -0.12061802, -0.40781283, -0.14290632, 0.013908465, 0.24071209, -0.10433602, 0.022023723, -0.092069246, 0.32779968, 0.21854237, 0.18298683, 0.049042679, -0.030719647, 0.10270946, 0.16853685, 0.10100971, -0.063925184], [-0.069782123, 0.1382505, 0.118492, -0.11041982, 0.12952189, -0.082892276, -0.048485439, 0.24909031, 0.049709395, 0.15676031, -0.28862923, 0.098646253, -0.48048675, -0.35209763, -0.098127969, -0.11421605, -0.57045311, -0.52655238, 0.17590456, 0.01918314, 0.048722688, -0.52502084, -0.47500119, 0.098212317, 0.06114829, 0.1697004, 0.063605741, 0.11854716, 0.059971351, -0.16208267, -0.51058233, -0.013609217], [0.12913682, 0.13424405, 0.34624648, -0.060588911, 0.035495538, 0.24723399, -0.15268147, -0.025497183, 0.18902484, 0.055687208, -0.11085252, 0.18597831, -0.076599233, -0.081077605, -0.01052121, -0.25128663, -0.058860218, -0.31528237, 0.09662468, -0.010629336, 0.24003923, -0.41454667, -0.24516213, 0.013819193, -0.027172125, 0.089294992, 0.18160732, 0.20986509, -0.074374399, -0.10950013, -0.3717941, 0.06230019], [-0.22456633, 0.066414878, -0.043317992, -0.32010928, -0.026089547, -0.11317018, 0.12341937, 0.18423273, -0.14123429, 0.15062009, -0.30966145, 0.11674522, -0.56669569, -0.62356365, -0.11015952, 0.1466991, -0.42896903, -0.059532162, 0.08451701, -0.014529812, 0.057836108, 0.27217281, 0.12306453, -0.012609994, 0.074989818, -0.073737122, 0.28622568, 0.27358148, 0.11629123, 0.11636252, -0.049600098, -0.11198422]]
ROOT_B2 = [0.13837707, 0.25362575, 0.021184525, -0.0078093545, 0.38106099, 0.16325532, 0.042178817, 0.41609707, -0.4935874, 0.083758451, 0.38134792, 0.52385587, 0.20238966, 0.41408584, 0.45067793, 0.39109546]
ROOT_W3 = [[0.25481233, -0.22720382, 0.21799341, 0.23091817, -0.11147328, -0.48968169, 0.13544695, -0.42882398, 0.46823156, -0.61088049, 0.20447521, 0.20254491, -0.59720832, -0.40107188, 0.23581544, -0.43612999], [-0.33504924, 0.115072, 0.20802478, 0.48491251, 0.29332206, 0.037042502, 0.26362073, -0.4645803, 0.32071453, 0.24127786, -0.62575817, -0.46747482, 0.29549411, -0.66442996, -0.47661522, 0.12416563], [-0.094169065, 0.28255364, -0.086916253, 0.28861105, 0.31119668, 0.14996223, -0.36468557, -0.30192935, 0.030813579, 0.53658211, -0.38332066, -0.58171624, 0.072992705, -0.64761019, -0.45281509, 0.25462493], [-0.037166402, -0.14741707, 0.55100316, -0.01331363, -0.053613074, -0.28122357, 0.25960791, -0.27805009, 0.54211545, 0.09982352, 0.036788095, 0.1635927, -0.1861756, -0.32346445, 0.034446992, -0.21814962], [0.30250579, -0.16450398, 0.099917255, -0.46115777, -0.18247128, -0.0038710376, 0.39617285, 0.12909718, -0.021365732, -0.19999753, 0.37205857, 0.074695468, -0.11816821, 0.22181021, 0.23521814, 0.12797315], [0.015020136, 0.30633488, 0.28666055, 0.29934105, 0.37415707, 0.30559775, -0.084982432, 0.15047464, -0.23279859, -0.25855324, 0.002383878, 0.23586619, 0.4245908, -0.0058383453, 0.011191974, -0.19281986], [-0.25351307, 0.16576469, 0.18142791, 0.42809382, 0.28728572, 0.23649363, -0.41132781, -0.42775035, -0.16978748, 0.24194518, -0.23638296, -0.4759272, 0.094906263, -0.18645915, -0.21971011, 0.16006751], [-0.60174489, 0.22393653, -0.13762872, -0.60289061, 0.26756102, 0.28921294, -0.11355869, 0.35655198, -0.33701, -0.33765426, 0.050916538, 0.1349836, 0.11056453, 0.40946543, 0.19087611, 0.35916635]]
ROOT_B3 = [0.15981825, 0.081263065, 0.23766172, 0.084207147, 0.31493694, 0.089028448, 0.17309405, 0.307107]
ROOT_W4 = [[0.43377125, 0.22068638, -0.20890656, 0.50805986, -0.0015690047, -0.17483254, -0.11265097, -0.28150034], [0.5478934, -0.87911314, -0.49133727, 0.28575245, 0.19226796, 0.20011559, -0.2155405, -0.025657903], [0.29289818, -0.48806936, -0.047623266, -0.15296604, 0.23992643, -0.67473459, -0.67935723, -0.23525529], [-0.080217645, -0.26308388, -0.038719043, -0.21475114, 0.23013303, 0.29234186, 0.0083044274, 0.27021432], [-0.21969169, -0.21932118, 0.14991829, -0.41066954, 0.10101501, 0.089791216, 0.19463135, 0.11738367], [-0.23226193, 0.064971849, 0.41095927, -0.8571378, -0.30907299, -0.079781614, 0.38068569, -6.4385458e-05], [0.14198524, 0.25927946, 0.47133574, 0.030559033, -0.23703276, 0.11295684, 0.56017172, -0.18223196]]
ROOT_B4 = [-0.047707886, -0.12322493, -0.27147678, -0.038782127, 0.17887489, 0.12706666, -0.42595062]


def predict_delta(product, x):
    if product == "INTARIAN_PEPPER_ROOT":
        logits = MLP_forward_logits(x, ROOT_MEAN, ROOT_SCALE,
                                    ROOT_W1, ROOT_B1, ROOT_W2, ROOT_B2,
                                    ROOT_W3, ROOT_B3, ROOT_W4, ROOT_B4)
        probs = softmax(logits)
        cls = argmax_index(logits)
        return ROOT_CLASS_VALUES[cls], probs[cls], cls, probs
    return 0.0, 0.0, -1, []


def extract_features(od: OrderDepth, market_trades, trader_state, product):
    bid_price_1 = float(max(od.buy_orders.keys())) if od.buy_orders else 0.0
    ask_price_1 = float(min(od.sell_orders.keys())) if od.sell_orders else 0.0
    bid_volume_1 = float(od.buy_orders.get(bid_price_1, 0)) if od.buy_orders else 0.0
    ask_volume_1 = float(abs(od.sell_orders.get(ask_price_1, 0))) if od.sell_orders else 0.0

    if bid_price_1 <= 0 or ask_price_1 <= 0:
        return [0.0] * 18, 0.0, trader_state

    mid = 0.5 * (bid_price_1 + ask_price_1)
    trades = market_trades.get(product, [])
    trade_count = float(len(trades))
    trade_qty = float(sum(abs(t.quantity) for t in trades))
    trade_vwap = (sum(float(t.price) * abs(t.quantity) for t in trades) / trade_qty
                  if trade_qty > 0 else mid)

    spread = ask_price_1 - bid_price_1
    imbalance = bid_volume_1 - ask_volume_1
    total_vol_1 = bid_volume_1 + ask_volume_1
    imbalance_ratio = imbalance / (total_vol_1 + 1e-6) if total_vol_1 > 0 else 0.0
    microprice = ((ask_price_1 * bid_volume_1 + bid_price_1 * ask_volume_1) / (total_vol_1 + 1e-6)
                  if total_vol_1 > 0 else mid)
    micro_diff = microprice - mid
    mid_minus_vwap = mid - trade_vwap

    sorted_bids = sorted(od.buy_orders.keys(), reverse=True)[:3]
    sorted_asks = sorted(od.sell_orders.keys())[:3]
    total_bid_vol_3 = float(sum(float(od.buy_orders[p]) for p in sorted_bids))
    total_ask_vol_3 = float(sum(float(abs(od.sell_orders[p])) for p in sorted_asks))
    book_total_3 = total_bid_vol_3 + total_ask_vol_3
    book_imbalance_3 = ((total_bid_vol_3 - total_ask_vol_3) / (book_total_3 + 1e-6)
                        if book_total_3 > 0 else 0.0)

    prev_key = f"prev_{product}"
    prev = trader_state.get(prev_key, {})
    prev_mid = float(prev.get("mid", mid))
    prev_spread = float(prev.get("spread", spread))
    prev_imbalance_ratio = float(prev.get("imbalance_ratio", imbalance_ratio))
    prev_mid_change = mid - prev_mid

    trader_state[prev_key] = {"mid": mid, "spread": spread, "imbalance_ratio": imbalance_ratio}

    features = [
        bid_price_1, bid_volume_1, ask_price_1, ask_volume_1,
        trade_count, trade_qty, trade_vwap,
        spread, imbalance, imbalance_ratio,
        micro_diff, mid_minus_vwap,
        total_bid_vol_3, total_ask_vol_3, book_imbalance_3,
        prev_mid_change, prev_spread, prev_imbalance_ratio,
    ]
    return features, mid, trader_state


# =============================================================================
#  LOGGER
# =============================================================================

class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict, conversions: int, trader_data: str) -> None:
        base_length = len(self.to_json([
            self.compress_state(state, ""),
            self.compress_orders(orders), conversions, "", "",
        ]))
        max_item_length = (self.max_log_length - base_length) // 3
        print(self.to_json([
            self.compress_state(state, self.truncate(state.traderData, max_item_length)),
            self.compress_orders(orders),
            conversions,
            self.truncate(trader_data, max_item_length),
            self.truncate(self.logs, max_item_length),
        ]))
        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list:
        return [state.timestamp, trader_data,
                self.compress_listings(state.listings),
                self.compress_order_depths(state.order_depths),
                self.compress_trades(state.own_trades),
                self.compress_trades(state.market_trades),
                state.position,
                self.compress_observations(state.observations)]

    def compress_listings(self, listings):
        return [[l.symbol, l.product, l.denomination] for l in listings.values()]

    def compress_order_depths(self, order_depths):
        return {s: [od.buy_orders, od.sell_orders] for s, od in order_depths.items()}

    def compress_trades(self, trades):
        return [[t.symbol, t.price, t.quantity, t.buyer, t.seller, t.timestamp]
                for arr in trades.values() for t in arr]

    def compress_observations(self, observations):
        conv_obs = {}
        for product, obs in observations.conversionObservations.items():
            conv_obs[product] = [obs.bidPrice, obs.askPrice, obs.transportFees,
                                 obs.exportTariff, obs.importTariff, obs.sugarPrice, obs.sunlightIndex]
        return [observations.plainValueObservations, conv_obs]

    def compress_orders(self, orders):
        return [[o.symbol, o.price, o.quantity] for arr in orders.values() for o in arr]

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        lo, hi = 0, min(len(value), max_length)
        out = ""
        while lo <= hi:
            mid = (lo + hi) // 2
            candidate = value[:mid]
            if len(candidate) < len(value):
                candidate += "..."
            if len(json.dumps(candidate)) <= max_length:
                out = candidate
                lo = mid + 1
            else:
                hi = mid - 1
        return out


logger = Logger()


# =============================================================================
#  CONFIGURATION
# =============================================================================

POSITION_LIMITS = {
    "ASH_COATED_OSMIUM":    80,
    "INTARIAN_PEPPER_ROOT": 80,
}

SESSION_END_TIMESTAMP = 1_000_000
EOD_FLUSH_WINDOW      = 50_000

# ── MARKET ACCESS FEE ────────────────────────────────────────────────────────
# Bid is in XIRECs (same unit as PnL). The briefing examples (10-34) are
# illustrative small numbers, not indicative of competition bids.
# Extra flow value ~20-25k XIRECs; 8k clears median under most distributions.
MAF_BID = 8_000

# ── OSMIUM ────────────────────────────────────────────────────────────────────
# MM_HALF=5 confirmed as R2 backtester sweep winner (+44% vs MM_HALF=7).
# LADDER_SPLIT=0.8 concentrates 80% budget at primary (tighter) quote.
# LADDER_OFFSET=2 puts secondary quote only 2 ticks behind primary.
# NPC book spread is ~16 ticks (bid1 ~9992, ask1 ~10008); our primary at
# ±5 sits well inside that, earning consistent edge on directional moves.
OSMIUM_FAIR            = 10_000
OSMIUM_MM_HALF         = 5
OSMIUM_MM_HALF_EOD     = 4
OSMIUM_TAKE_EDGE       = 1
OSMIUM_EXTREME_DEV     = 20
OSMIUM_MAX_SKEW        = 6
OSMIUM_SKEW_POWER      = 1.5
OSMIUM_LADDER_OFFSET   = 2
OSMIUM_LADDER_SPLIT    = 0.8
OSMIUM_UNWIND_THRESHOLD     = 30
OSMIUM_UNWIND_THRESHOLD_EOD = 20
OSMIUM_UNWIND_SACRIFICE     = 2
OSMIUM_UNWIND_SACRIFICE_EOD = 4
OSMIUM_UNWIND_FRACTION      = 0.5
OSMIUM_UNWIND_FRACTION_EOD  = 1.0

# ── PEPPER TREND ─────────────────────────────────────────────────────────────
# Slope = 1/1000 confirmed exactly across all 3 wiki data days.
# Residual std ≈ 2.5 ticks; worst single-tick drop = -18 ticks in the data.
#
# BUY_EDGE raised 8 → 9:
#   NPC asks shift from mid+6.5 (day -1) to mid+7.5 (day 1).
#   BUY_EDGE=8 fills 97.5% / 81.9% / 71.6% across the 3 days.
#   BUY_EDGE=9 fills 99.3% / 97.3% / 94.9% — recovers ~23% of missed fills
#   on day 1. Cost is 1 extra tick per buy, negligible vs 10k-tick trend gain.
PEPPER_SLOPE                     = 1.0 / 1000
PEPPER_BUY_EDGE                  = 9       # raised from 8 based on wiki data analysis
PEPPER_SELL_SPIKE                = 8
PEPPER_PASSIVE_BID_OFFSET        = 2
PEPPER_INTERCEPT_EWMA            = 0.0005
PEPPER_INTERCEPT_RESET_THRESHOLD = 300
PEPPER_MLP_CONFIDENCE            = 0.4
PEPPER_MLP_MAX_SHIFT             = 3

# ── PEPPER REGIME DETECTION ───────────────────────────────────────────────────
# Wiki data shows PEPPER essentially never crashes (residual always < ±12).
# These thresholds are therefore a safety net for unseen scenarios, not a
# mechanism that fires regularly. Kept conservative to avoid false triggers.
PEPPER_EWMA_RESIDUAL_ALPHA   = 0.08
PEPPER_EWMA_MOMENTUM_ALPHA   = 0.12
PEPPER_EWMA_SPREAD_ALPHA     = 0.05
PEPPER_BEARISH_RESIDUAL      = -12.0  # both required: prevents whipsawing
PEPPER_BEARISH_MOMENTUM      = -0.8
PEPPER_HOLD_RESIDUAL         = -6.0   # either suffices for HOLD
PEPPER_HOLD_MOMENTUM         = -0.3
PEPPER_BULLISH_RESIDUAL      = -2.0   # both required to exit HOLD/BEARISH
PEPPER_BULLISH_MOMENTUM      = 0.1
PEPPER_SPREAD_SHOCK_MULT     = 3.0
PEPPER_BEARISH_UNWIND_SACRIFICE = 4
PEPPER_BEARISH_MAX_UNWIND_QTY   = 10


# =============================================================================
#  TRADER
# =============================================================================

class Trader:

    # -------------------------------------------------------------------------
    #  ROUND 2: Market Access Fee bid
    #  Only the top 50% of bids win, paying their own bid amount.
    #  Winners receive 25% more order book flow for the full simulation.
    # -------------------------------------------------------------------------
    def bid(self):
        return MAF_BID

    def run(self, state: TradingState):
        result: Dict[Symbol, List[Order]] = {}
        trader_state = {}
        if state.traderData:
            try:
                trader_state = json.loads(state.traderData)
            except Exception:
                trader_state = {}

        for product in state.order_depths:
            od       = state.order_depths[product]
            position = state.position.get(product, 0)
            limit    = POSITION_LIMITS.get(product, 80)

            try:
                if product == "ASH_COATED_OSMIUM":
                    result[product] = self.trade_osmium(od, position, limit, state.timestamp)
                elif product == "INTARIAN_PEPPER_ROOT":
                    result[product], trader_state = self.trade_pepper(
                        od, position, limit, state.timestamp, trader_state, state.market_trades)
            except Exception as e:
                result[product] = []
                logger.print(f"ERR {product}: {e}")

        trader_data = json.dumps(trader_state)
        logger.flush(state, result, 0, trader_data)
        return result, 0, trader_data

    # =========================================================================
    #  ASH_COATED_OSMIUM
    #  Passive MM at fair=10000 with nonlinear inventory skew and EOD flush.
    #  MM_HALF=5 is the R2 sweep winner. Tight ladder (offset=2, split=0.8)
    #  puts 80% of capacity at the primary quote and 20% as a backstop.
    # =========================================================================

    def trade_osmium(self, od: OrderDepth, position: int, limit: int, timestamp: int) -> List[Order]:
        orders: List[Order] = []
        fair = OSMIUM_FAIR

        best_bid = max(od.buy_orders.keys()) if od.buy_orders else None
        best_ask = min(od.sell_orders.keys()) if od.sell_orders else None
        if best_bid is None and best_ask is None:
            return orders

        mid = ((best_bid + best_ask) / 2 if best_bid is not None and best_ask is not None
               else float(best_bid if best_bid is not None else best_ask))

        eod              = (SESSION_END_TIMESTAMP - timestamp) <= EOD_FLUSH_WINDOW
        mm_half          = OSMIUM_MM_HALF_EOD       if eod else OSMIUM_MM_HALF
        unwind_threshold = OSMIUM_UNWIND_THRESHOLD_EOD if eod else OSMIUM_UNWIND_THRESHOLD
        unwind_sacrifice = OSMIUM_UNWIND_SACRIFICE_EOD if eod else OSMIUM_UNWIND_SACRIFICE
        unwind_fraction  = OSMIUM_UNWIND_FRACTION_EOD  if eod else OSMIUM_UNWIND_FRACTION

        dev                  = mid - fair
        effective_take_edge  = 0 if abs(dev) > OSMIUM_EXTREME_DEV else OSMIUM_TAKE_EDGE

        # 1. Aggressive takes: buy cheap, sell rich vs fair value
        if od.sell_orders:
            for ask in sorted(od.sell_orders.keys()):
                if ask < fair - effective_take_edge:
                    qty = min(-od.sell_orders[ask], limit - position)
                    if qty > 0:
                        orders.append(Order("ASH_COATED_OSMIUM", ask, qty))
                        position += qty

        if od.buy_orders:
            for bid in sorted(od.buy_orders.keys(), reverse=True):
                if bid > fair + effective_take_edge:
                    qty = min(od.buy_orders[bid], limit + position)
                    if qty > 0:
                        orders.append(Order("ASH_COATED_OSMIUM", bid, -qty))
                        position -= qty

        # 2. Inventory unwind when position becomes extreme
        if position > unwind_threshold and od.buy_orders:
            target = int((position - unwind_threshold) * unwind_fraction)
            for bid_px in sorted(od.buy_orders.keys(), reverse=True):
                if target <= 0: break
                if bid_px < fair - unwind_sacrifice: break
                if bid_px > fair + effective_take_edge: continue
                qty = min(od.buy_orders[bid_px], target, limit + position)
                if qty > 0:
                    orders.append(Order("ASH_COATED_OSMIUM", bid_px, -qty))
                    position -= qty
                    target   -= qty

        elif position < -unwind_threshold and od.sell_orders:
            target = int((-position - unwind_threshold) * unwind_fraction)
            for ask_px in sorted(od.sell_orders.keys()):
                if target <= 0: break
                if ask_px > fair + unwind_sacrifice: break
                if ask_px < fair - effective_take_edge: continue
                qty = min(-od.sell_orders[ask_px], target, limit - position)
                if qty > 0:
                    orders.append(Order("ASH_COATED_OSMIUM", ask_px, qty))
                    position += qty
                    target   -= qty

        # 3. Passive MM with nonlinear inventory skew
        if limit > 0:
            inv_ratio    = abs(position) / limit
            inv_skew_mag = (inv_ratio ** OSMIUM_SKEW_POWER) * OSMIUM_MAX_SKEW
            inv_skew     = -inv_skew_mag if position > 0 else inv_skew_mag
        else:
            inv_skew = 0.0

        skew     = max(-OSMIUM_MAX_SKEW, min(OSMIUM_MAX_SKEW, inv_skew))
        bid_px_1 = int(round(fair - mm_half + skew))
        ask_px_1 = int(round(fair + mm_half + skew))
        bid_px_1 = min(bid_px_1, fair - 1)
        ask_px_1 = max(ask_px_1, fair + 1)
        bid_px_2 = bid_px_1 - OSMIUM_LADDER_OFFSET
        ask_px_2 = ask_px_1 + OSMIUM_LADDER_OFFSET

        bid_budget = limit - position
        ask_budget = limit + position

        if bid_budget > 0:
            bid_qty_1 = int(round(bid_budget * OSMIUM_LADDER_SPLIT))
            bid_qty_2 = bid_budget - bid_qty_1
            if bid_qty_1 > 0:
                orders.append(Order("ASH_COATED_OSMIUM", bid_px_1, bid_qty_1))
            if bid_qty_2 > 0:
                orders.append(Order("ASH_COATED_OSMIUM", bid_px_2, bid_qty_2))

        if ask_budget > 0:
            ask_qty_1 = int(round(ask_budget * OSMIUM_LADDER_SPLIT))
            ask_qty_2 = ask_budget - ask_qty_1
            if ask_qty_1 > 0:
                orders.append(Order("ASH_COATED_OSMIUM", ask_px_1, -ask_qty_1))
            if ask_qty_2 > 0:
                orders.append(Order("ASH_COATED_OSMIUM", ask_px_2, -ask_qty_2))

        return orders

    # =========================================================================
    #  INTARIAN_PEPPER_ROOT
    #  Regime-aware trend rider. Slope = 1/1000 confirmed on all 3 data days.
    #
    #  BULLISH  — buy aggressively up to max long; BUY_EDGE=9 captures 95%+
    #             of NPC asks across all days (vs 72% at BUY_EDGE=8 on day 1).
    #             Scratch-sell trims top of position when well above trend.
    #
    #  HOLD     — no new buys; hold position; light ask at fair+3 earns edge.
    #
    #  BEARISH  — controlled unwind, max 10 units/tick, floor at fair-4.
    #             Both EWMA residual AND momentum must fire to enter BEARISH
    #             (avoids false triggers on noise; wiki data residual std = 2.5).
    # =========================================================================

    def trade_pepper(self, od: OrderDepth, position: int, limit: int,
                     timestamp: int, trader_state: dict, market_trades) -> tuple:
        orders: List[Order] = []

        best_bid = max(od.buy_orders.keys()) if od.buy_orders else None
        best_ask = min(od.sell_orders.keys()) if od.sell_orders else None
        if best_bid is None and best_ask is None:
            return orders, trader_state

        mid    = ((best_bid + best_ask) / 2 if best_bid is not None and best_ask is not None
                  else float(best_bid if best_bid is not None else best_ask))
        spread = (best_ask - best_bid) if (best_bid is not None and best_ask is not None) else 10

        # 1. Bootstrap / maintain trend intercept
        expected_intercept = mid - timestamp * PEPPER_SLOPE
        pepper_key         = "pepper_intercept"
        need_bootstrap     = (
            pepper_key not in trader_state
            or not isinstance(trader_state.get(pepper_key), (int, float))
            or abs(trader_state[pepper_key] - expected_intercept) > PEPPER_INTERCEPT_RESET_THRESHOLD
        )
        if need_bootstrap:
            trader_state[pepper_key] = expected_intercept

        intercept  = trader_state[pepper_key]
        trend_fair = intercept + timestamp * PEPPER_SLOPE
        residual   = mid - trend_fair
        trader_state[pepper_key] = intercept + PEPPER_INTERCEPT_EWMA * residual

        # 2. Regime signal EWMAs
        prev_mid_key = "pepper_prev_mid"
        prev_mid_val = trader_state.get(prev_mid_key, mid)
        momentum     = mid - prev_mid_val
        trader_state[prev_mid_key] = mid

        ewma_res_key = "pepper_ewma_res"
        ewma_res     = trader_state.get(ewma_res_key, 0.0)
        ewma_res     = (1 - PEPPER_EWMA_RESIDUAL_ALPHA) * ewma_res + PEPPER_EWMA_RESIDUAL_ALPHA * residual
        trader_state[ewma_res_key] = ewma_res

        ewma_mom_key = "pepper_ewma_mom"
        ewma_mom     = trader_state.get(ewma_mom_key, 0.0)
        ewma_mom     = (1 - PEPPER_EWMA_MOMENTUM_ALPHA) * ewma_mom + PEPPER_EWMA_MOMENTUM_ALPHA * momentum
        trader_state[ewma_mom_key] = ewma_mom

        ewma_spd_key = "pepper_ewma_spd"
        ewma_spd     = trader_state.get(ewma_spd_key, float(spread))
        ewma_spd     = (1 - PEPPER_EWMA_SPREAD_ALPHA) * ewma_spd + PEPPER_EWMA_SPREAD_ALPHA * spread
        trader_state[ewma_spd_key] = ewma_spd

        vol_shock = spread > PEPPER_SPREAD_SHOCK_MULT * ewma_spd

        # 3. Regime state machine (0=BULLISH, 1=HOLD, 2=BEARISH)
        regime_key = "pepper_regime"
        regime     = trader_state.get(regime_key, 0)

        if regime == 0:    # BULLISH
            if ewma_res < PEPPER_BEARISH_RESIDUAL and ewma_mom < PEPPER_BEARISH_MOMENTUM:
                regime = 2
            elif ewma_res < PEPPER_HOLD_RESIDUAL or ewma_mom < PEPPER_HOLD_MOMENTUM:
                regime = 1
        elif regime == 1:  # HOLD
            if ewma_res < PEPPER_BEARISH_RESIDUAL and ewma_mom < PEPPER_BEARISH_MOMENTUM:
                regime = 2
            elif ewma_res > PEPPER_BULLISH_RESIDUAL and ewma_mom > PEPPER_BULLISH_MOMENTUM:
                regime = 0
        elif regime == 2:  # BEARISH
            if ewma_res > PEPPER_BULLISH_RESIDUAL and ewma_mom > PEPPER_BULLISH_MOMENTUM:
                regime = 0

        trader_state[regime_key] = regime

        # 4. MLP fair-value enhancement (BULLISH regime + high confidence only)
        fair = trend_fair
        if regime == 0:
            try:
                x, mlp_mid, trader_state = extract_features(
                    od, market_trades, trader_state, "INTARIAN_PEPPER_ROOT")
                if mlp_mid > 0:
                    delta, confidence, cls, probs = predict_delta("INTARIAN_PEPPER_ROOT", x)
                    if confidence > PEPPER_MLP_CONFIDENCE:
                        shift = max(-PEPPER_MLP_MAX_SHIFT,
                                    min(PEPPER_MLP_MAX_SHIFT, delta * 0.5))
                        fair = trend_fair + shift
            except Exception:
                fair = trend_fair

        regime_label = ["BULLISH", "HOLD", "BEARISH"][regime]
        logger.print(f"P t={timestamp} {regime_label} fair={fair:.1f} "
                     f"res={ewma_res:.2f} mom={ewma_mom:.3f} pos={position}")

        # 5. Order generation by regime

        if regime == 0:  # BULLISH — maximise long position
            # 5a. Aggressive takes: buy asks ≤ fair + BUY_EDGE=9
            #     Captures 95%+ of NPC asks on all wiki data days.
            if od.sell_orders and not vol_shock:
                for ask in sorted(od.sell_orders.keys()):
                    if ask <= fair + PEPPER_BUY_EDGE:
                        qty = min(-od.sell_orders[ask], limit - position)
                        if qty > 0:
                            orders.append(Order("INTARIAN_PEPPER_ROOT", ask, qty))
                            position += qty

            # 5b. Spike sell: only dump if massively above trend
            if od.buy_orders and position > 0:
                for bid in sorted(od.buy_orders.keys(), reverse=True):
                    if bid > fair + PEPPER_SELL_SPIKE:
                        qty = min(od.buy_orders[bid], position)
                        if qty > 0:
                            orders.append(Order("INTARIAN_PEPPER_ROOT", bid, -qty))
                            position -= qty

            # 5c. Scratch-sell at max position when well above trend
            #     Locks in realised PnL without abandoning drift exposure.
            #     Position immediately refills via passive bid below.
            if od.buy_orders and position >= 78 and residual >= 4:
                scratch_remaining = 3
                for bid in sorted(od.buy_orders.keys(), reverse=True):
                    if scratch_remaining <= 0: break
                    if bid < fair + 4: break
                    if bid > fair + PEPPER_SELL_SPIKE: continue
                    qty = min(od.buy_orders[bid], scratch_remaining, position)
                    if qty > 0:
                        orders.append(Order("INTARIAN_PEPPER_ROOT", bid, -qty))
                        position          -= qty
                        scratch_remaining -= qty

            # 5d. Passive bid to fill remaining capacity
            remaining = limit - position
            if remaining > 0 and not vol_shock:
                passive_bid = int(round(fair + PEPPER_PASSIVE_BID_OFFSET))
                if best_ask is not None:
                    passive_bid = min(passive_bid, best_ask - 1)
                orders.append(Order("INTARIAN_PEPPER_ROOT", passive_bid, remaining))

        elif regime == 1:  # HOLD — no buying; earn edge on spikes
            if od.buy_orders and position > 0:
                for bid in sorted(od.buy_orders.keys(), reverse=True):
                    if bid > fair + PEPPER_SELL_SPIKE:
                        qty = min(od.buy_orders[bid], position)
                        if qty > 0:
                            orders.append(Order("INTARIAN_PEPPER_ROOT", bid, -qty))
                            position -= qty

            # Light ask at fair+3: inside NPC spread, earns edge while waiting
            if best_bid is not None and best_ask is not None and position > 0:
                hold_ask_px = int(round(fair + 3))
                ask_budget  = position
                if ask_budget > 0:
                    orders.append(Order("INTARIAN_PEPPER_ROOT", hold_ask_px, -min(ask_budget, 5)))

        elif regime == 2:  # BEARISH — controlled unwind, no panic selling
            if od.buy_orders and position > 0:
                unwind_target = min(PEPPER_BEARISH_MAX_UNWIND_QTY, position)
                floor_price   = int(round(fair - PEPPER_BEARISH_UNWIND_SACRIFICE))
                for bid in sorted(od.buy_orders.keys(), reverse=True):
                    if unwind_target <= 0: break
                    if bid < floor_price: break
                    qty = min(od.buy_orders[bid], unwind_target, position)
                    if qty > 0:
                        orders.append(Order("INTARIAN_PEPPER_ROOT", bid, -qty))
                        position      -= qty
                        unwind_target -= qty

        return orders, trader_state
