# paper_lsi_model.py output

2017-04-30 19:38:34,182 : INFO : adding document #0 to Dictionary(0 unique tokens: [])
2017-04-30 19:39:07,275 : INFO : built Dictionary(123965 unique tokens: [u'', u'bshouty', u'photometry', u'q-1', u'weights/synaptic']...) from 6560 documents (total 18865276 corpus positions)
2017-04-30 19:39:38,790 : INFO : collecting document frequencies
2017-04-30 19:39:38,790 : INFO : PROGRESS: processing document #0
2017-04-30 19:39:40,963 : INFO : calculating IDF weights for 6560 documents and 123964 features (4934951 matrix non-zeros)
2017-04-30 19:39:41,164 : INFO : using serial LSI version on this node
2017-04-30 19:39:41,165 : INFO : updating model with new documents
2017-04-30 19:39:54,678 : INFO : preparing a new chunk of documents
2017-04-30 19:39:56,721 : INFO : using 100 extra samples and 2 power iterations
2017-04-30 19:39:56,721 : INFO : 1st phase: constructing (123965, 120) action matrix
2017-04-30 19:39:59,130 : INFO : orthonormalizing (123965, 120) action matrix
2017-04-30 19:40:51,461 : INFO : 2nd phase: running dense svd on (120, 6560) matrix
2017-04-30 19:40:52,523 : INFO : computing the final decomposition
2017-04-30 19:40:52,528 : INFO : keeping 20 factors (discarding 51.233% of energy spectrum)
2017-04-30 19:40:53,096 : INFO : processed documents up to #6560
2017-04-30 19:40:53,201 : INFO : topic #0(12.553): 0.141*"neuron" + 0.115*"im" + 0.115*"''" + 0.113*"'" + 0.110*"kernel" + 0.103*"spik" + 0.099*"w" + 0.097*"clust" + 0.095*"network" + 0.080*"
nod"2017-04-30 19:40:53,219 : INFO : topic #1(7.633): -0.459*"neuron" + -0.394*"spik" + -0.170*"cel" + -0.164*"stimul" + -0.160*"fir" + -0.146*"synapt" + -0.146*"synaps" + -0.100*"circuit" + 0.09
8*"kernel" + -0.097*"inhibit"2017-04-30 19:40:53,226 : INFO : topic #2(6.290): -0.359*"policy" + -0.231*"" + 0.205*"im" + -0.200*"reward" + -0.191*"" + -0.188*"" + -0.175*"spik" + -0.171*"regret" + -0.165*"" + -0.164*""
Xshell2017-04-30 19:40:53,233 : INFO : topic #3(6.193): 0.377*"policy" + -0.287*"" + -0.232*"" + -0.227*"" + 0.223*"reward" + -0.204*"" + -0.203*"" + -0.194*" + 0.186*"regret" + -0.171*""
2017-04-30 19:40:53,240 : INFO : topic #4(5.961): -0.340*"spik" + 0.317*"policy" + -0.206*"neuron" + 0.188*"''" + 0.165*"'" + -0.152*"kernel" + 0.149*"reward" + 0.139*"im" + 0.130*"``" + 0.11
7*"lay"Xshell2017-04-30 19:40:53,705 : INFO : saving Projection object under lsi_model.projection, separately None
2017-04-30 19:40:53,824 : INFO : saved lsi_model.projection
2017-04-30 19:40:53,825 : INFO : saving LsiModel object under lsi_model, separately None
2017-04-30 19:40:53,825 : INFO : not storing attribute projection
2017-04-30 19:40:53,825 : INFO : not storing attribute dispatcher
2017-04-30 19:40:54,318 : INFO : saved lsi_model

# paper_sim_measure.py output

Paper Title:  ['Plasticity-Mediated Competitive Learning']
Similar Papers:  ['Plasticity-Mediated Competitive Learning']
Similarity Rate:  1.0
Similar Papers:  ['Self Organizing Neural Networks for the Identification Problem']
Similarity Rate:  0.987902
Similar Papers:  ['Spreading Activation over Distributed Microfeatures']
Similarity Rate:  0.969999
Similar Papers:  ['The Storage Capacity of a Fully-Connected Committee Machine']
Similarity Rate:  0.967154
Similar Papers:  ['Time Trials on Second-Order and Variable-Learning-Rate Algorithms']
Similarity Rate:  0.965807
Similar Papers:  [ 'The Boltzmann Perceptron Network: A Multi-Layered Feed-Forward Network Equivalent to the Boltzmann Machine']
Similarity Rate:  0.965145
Similar Papers:  ['Microscopic Equations in Rough Energy Landscape for Neural Networks']
Similarity Rate:  0.964354
Similar Papers:  [ 'S-Map: A Network with a Simple Self-Organization Algorithm for Generative Topographic Mappings']
Similarity Rate:  0.961203


