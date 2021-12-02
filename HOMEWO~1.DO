/* 
		 ********************************************
		 ** EMPIRICAL APPLICATION - CREDIT SCORING **
		 ********************************************
		 
Authors : 
	- Roland BOUILLOT
	- Khalil JANBEK
	- Mehdi LOUAFI
	
Dofile based on the one proposed by JB Chatelain (5 november 2017)

Data file used in this application: defaut2000.dta

Programme de prédiction de faillite: Binary dependent variable     
181 entreprises manufacturières dont 86 en difficulté et 95 en bonne santé 
cotées sur le New York Stock Exchange ou AMEX Compustat, en coupe 
mais annees differentes suivant entreprises: 1981 ‡ 1989.
Etude de Theodossiou Kahya Saidi Philippatos (TKSP) 1996
Journal of Business Finance and Accounting
Donnees p.717
Leur modele: choisi sur minimisation du critère d'Akaike (AIC). 


		************* Summary *************

		Chapter 1 : Estimation Sample 
			- Part I   : Data cleaning		
			- Part II  : Questions 1 to 18
			
		Chapter 2 : Validation Sample
			- Part I   : Data cleaning		
			- Part II  : Questions 19 to 21
		
		***********************************
	

*/

		*********************************
		* CHAPTER I : ESTIMATION SAMPLE *
		*********************************

{
********** SETUP **********
clear all
set more off
version 14.0
set linesize 90
set scheme s1mono

* install packages
ssc install asdoc
ssc install spost9_ado
ssc install cutpt 

* Directory 
cd "C:\~\Homework Credit Scoring - Bouillot Janbek Louafi\Code"

* Log capture
capture log close
log using defaut-v1.txt, text replace

* Open data file
* Remark: Data are already sorted by yd tdta 
use "C:\~\Homework Credit Scoring - Bouillot Janbek Louafi\Code\defaut2000bis.dta", clear

* Proc content
des

		******************************
		* Part I : Cleaning the Data *
		******************************

	
	* STEP 1: Labeling our data 
	***************************
	
label variable yd      "Indic Difficultés Financières" 
label variable tdta    "Dette/Actif"   /* total debt/total assets */
label variable reta    "Reserves et report ‡ nouveau/Actif" /* retained earnings */
label variable opita   "Resultat/Actif"  /* operating income/total assets */
label variable ebita   "EBIT ou EBE/Actif" /* earnings before interest taxes/total assets*/
label variable lsls    "Log(ventes)" /* log sales */
label variable lta     "Log(actif)"  /* log assets */
label variable gempl   "Croissance du nombre de salariés" /* growth of employment */
label variable invsls  "Stocks/Ventes"  /* inventories/sales */
label variable nwcta   "Capital circulant net/Actif" /* net working capital/total assets */
label variable cacl    "Actif circulant/Passif circulant" /* current assets/current liabilities */
label variable qacl    "Disponibilites+VMP/Passif circulant" /* quick assets/current liabilities */ 
label variable fata    "Actifs Fixes/Actif" /* fixed assets/total assets */
label variable ltdta   "Dette Long Terme/Actif" /* long term debt/total assets */
label variable mveltd  "Valeur de marche Actions/Dette long terme" /* market value equity/lgterm debt */

label define ydl 0 "No Fin.Dis." 1 "Fin.Dist"
label values yd ydl

* Proc labeled content
describe

* Create an xlist string for the list of regressors (called by $xlist)
global xlist tdta reta opita ebita lsls lta gempl invsls nwcta cacl qacl fata ltdta mveltd

* Tables 
table yd
tabulate yd

tdta reta lsls gempl nwcta mveltd

	* STEP 2: Sort and clean data 
	*****************************

* Sorting data by yd and ebita
sort yd ebita

* Cleaning data (by replacing the -99.98 observations by a "missing obs.")
replace fata = . if fata <= -99.98
replace ltdta = . if ltdta <= -99.98


	* STEP 3: Creating the ESTIMATION sample
	****************************************
	
*list yd $xlist if mod(_n,2) 
*list yd $xlist if !mod(_n,2)

* Creating the estimation sample (validation sample is created in line XXX)
drop if !mod(_n,2)
describe

* Save Clean dataset for Estimation
save CleanDataEstimation.dta, replace
clear all

****************************************************************************************************
****************************************************************************************************


		*******************************
		* Part II : Questions 1 to 18 *
		*******************************
		
* We use the Clean data validation dataset
clear all
u "CleanDataEstimation.dta"

	** Question 1 ** 
	****************

* No code needed, answer available in the report


	** Question 2 **
	****************

* Univariate statistics on overall distribution
* Full sample: N, mean, median, skewness, kurtosis, min, max, percentiles, standard deviation, variance
tabstat yd $xlist, stat (count mean p50 skew min p5 p25 p75 p95 max kurt iqr sd variance ) col(stat)


	** Question 3 **
	****************

* Summary statistics and histogram of total debt/total assets (tdta) for default versus healthy firms (yd)
* If p-value<5%, we reject normality
asdoc sum yd tdta ebita, save(CreditScoringTables) replace
histogram tdta, bin(10) color(white) lcolor(green) kdensity kdenopts(lcolor(blue) lwidth(thick) ) normal normopts(lcolor(red) lwidth(thick) ) by(yd, total holes(2))


	** Question 4 **
	****************

/*************************************************
 L'analyse discriminante se fait sous hypothese de multinormalité de l'ensemble
des variables pour chaque groupe (avec ou sans hypothesede matrice 
de variance covariance égale entre les 2 groupes).
L'Èconométrie de base fait l'hypothese que les variables sont non aléatoires
donc pas necessaire de vérifier la normalité des observations des variables, 
sauf si on présume endogénéité et erreur de mesure, cf. variables instrumentales.
******************************************/

* Normality tests for total debt/total assets (tdta) for default versus healthy firms (yd)

	* Standardized normal probability plot
pnorm tdta if yd==0
pnorm tdta if yd==1

	* Quantile–normal plot
qnorm tdta if yd==0
qnorm tdta if yd==1

	*Skewness/Kurtosis tests
asdoc sktest tdta if yd==0, save(CreditScoringTables) replace
asdoc sktest tdta if yd==1, save(CreditScoringTables) replace

	* Shapiro-Wilk W test
asdoc swilk tdta if yd==0, save(CreditScoringTables) replace
asdoc swilk tdta if yd==1, save(CreditScoringTables) replace

	* Multivariate normality test (Doornik-Hansen)
asdoc mvtest normality tdta reta if yd==1, save(CreditScoringTables) replace

	
	** Question 5 **
	****************

* 4 ways to get t=5.72 with p=0.000 and same R2=26.91% : 
	* (1) Simple correlation=0 test, 
	* (2) Analysis of variance, 
	* (3) Linear probability model, 
	* (4) Test of diff(means)

* Test 1 : T-test Difference of Means (if two normal distributions)
ttest tdta, by(yd) 
ttest ebita, by(yd) 

* Test 2 : Analysis of Variance
anova tdta yd
asdoc reg, save(CreditScoringTables) replace

* Test 3 : Linear probability model
asdoc reg yd tdta, save(CreditScoringTables) replace

* Test 4 : Simple correlation test
asdoc pwcorr yd tdta , sig obs star(0.05) save(CreditScoringTables) replace


	** Question 6 **
	****************

*  Box-plots of each of the two groups of firms for the 14 financial ratios
graph box tdta reta opita ebita gempl , over(yd) 
graph box invsls nwcta mveltd, over (yd)
graph box cacl qacl , over (yd)
graph box lsls lta , over (yd)
graph box ltdta fata , over (yd)


	** Question 7 **
	****************

* Bivariate correlations with the dependent variable
asdoc pwcorr yd tdta reta opita ebita gempl invsls nwcta mveltd  cacl qacl lsls lta ltdta fata , sig star(0.05) save(CreditScoringTables) replace
graph matrix yd tdta reta opita ebita gempl invsls nwcta mveltd  cacl qacl lsls lta ltdta fata 
graph matrix yd tdta reta opita ebita gempl 
graph matrix yd invsls nwcta mveltd  cacl qacl 
graph matrix yd lsls lta ltdta fata 

* Regressors which correlation is > |0.3| with the dependent variable
asdoc pwcorr yd tdta reta opita ebita ltdta, sig obs star(0.05) save(CreditScoringTables) replace
graph matrix yd tdta reta opita ebita ltdta 


	** Question 8 **
	****************
	
* Bivariate correlations between regressors
asdoc pwcorr tdta reta opita ebita gempl invsls nwcta mveltd cacl qacl lsls lta ltdta fata , sig star(0.05) save(CreditScoringTables) replace

* Regressors which correlation is > |0.3| with each other
asdoc pwcorr tdta reta opita ebita nwcta cacl qacl ltdta fata, sig obs star(0.05) save(CreditScoringTables) replace
graph matrix tdta reta opita ebita nwcta cacl qacl ltdta fata
	
	
	** Question 9 **
	****************

* ltda a la place de gempl ? Correlation plus élevée (-.30 vs -.27)
* Redondance avec question 8
asdoc pwcorr yd tdta reta opita ebita gempl, sig obs star(0.05) save(CreditScoringTables) replace
graph matrix yd tdta reta opita ebita gempl


	** Question 10 **
	*****************
	
* Deja présenté dans la question 5
asdoc reg yd tdta, save(CreditScoringTables) replace

* PBM1: predicted proba outside the interval [0 1]
graph twoway lfit   yd tdta || scatter yd tdta
graph twoway lfitci yd tdta || scatter yd tdta

rvfplot, yline(0)
predict lpredtdta
predict etdta, resid
label variable etdta "residuals tdta"
graph twoway lfit etdta tdta || scatter etdta tdta
graph twoway scatter yd etdta
	
	
	** Question 11 **
	*****************

* PBM2: Non-normality of residual of LPM OLS regression : 2 methods

* Normality tests
asdoc sktest etdta, save(CreditScoringTables) replace
asdoc swilk etdta, save(CreditScoringTables) replace

* Figure 1
histogram etdta, bin(10) color(white) lcolor(green) kdensity kdenopts(lcolor(blue) lwidth(thick) ) normal normopts(lcolor(red) lwidth(thick) ) 

* Figure 2 
histogram etdta, bin(10) color(white) lcolor(green) kdensity kdenopts(lcolor(blue) lwidth(thick) ) normal normopts(lcolor(red) lwidth(thick) ) by(yd, total holes(2))
graph box etdta , over(yd)  
graph twoway scatter yd etdta
graph twoway scatter etdta yd
histogram etdta, bin(10) color(white) lcolor(green) kdensity kdenopts(lcolor(blue) lwidth(thick) ) normal normopts(lcolor(red) lwidth(thick) ) by(yd, rows(2) )

	
	** Question 12 **
	*****************

* The logit model
asdoc logit yd tdta, save(CreditScoringTables) replace
fitstat

	* Distribution of Logit residuals

* Logit predicted (standardized) Pearson residuals
predict residualslog, residuals
predict rstandardlog, rstandard

histogram residualslog, bin(10) color(white) lcolor(green) kdensity kdenopts(lcolor(blue) lwidth(thick) ) normal normopts(lcolor(red) lwidth(thick) ) 
histogram rstandardlog, bin(10) color(white) lcolor(green) kdensity kdenopts(lcolor(blue) lwidth(thick) ) normal normopts(lcolor(red) lwidth(thick) ) 
histogram rstandardlog, by(yd)  bin(10) color(white) lcolor(green) kdensity kdenopts(lcolor(blue) lwidth(thick) ) normal normopts(lcolor(red) lwidth(thick) ) 

graph box residualslog
graph box rstandardlog

graph box residualslog, by(yd)
graph box rstandardlog, by(yd)


* Normality tests
sktest residualslog
swilk residualslog

asdoc sktest rstandardlog, save(CreditScoringTables) replace
asdoc swilk rstandardlog, save(CreditScoringTables) replace


	** Question 13 **
	*****************
	
* Comparing the OLS, Logit and Probit estimations	

* Beta OLS estimate 
reg yd tdta
estimates store bols

* Beta OLS estimate (robust)
reg yd tdta, vce(robust)
estimates store bolsr

* Beta logit estimate 
logit yd tdta
estimates store blogit 

* Beta logit estimate robust
logit yd tdta, vce(robust)
estimates store blogitr 

* Beta probit estimate 
probit yd tdta
estimates store bprobit

* Beta probit estimate robust
probit yd tdta, vce(robust)
estimates store bprobitr

* Beta estimates of OLS, Logit and Probit regressions (including robust estimates) 
asdoc estimates table bols bolsr bprobit bprobitr blogit blogitr , t stats(N ll)  b(%7.3f) stfmt(%8.2f), save(CreditScoringTables) replace

* Detailed results of the models
asdoc reg yd tdta, save(CreditScoringTables) replace
asdoc logit yd tdta, save(CreditScoringTables) replace
asdoc probit yd tdta, save(CreditScoringTables) replace


	** Question 14 **
	*****************

* No code needed, answer available in the report

	
	** Question 15 **
	*****************

**** Determining cutoff
asdoc cutpt yd tdta, noadjust save(CreditScoringTables) replace
roctab yd tdta, graph msymbol(none) addplot(scatteri `e(sens)' `=1 - e(spec)') legend(label(3 "Cutpoint"))
bootstrap e(cutpoint), rep(100): cutpt yd tdta, noadjust 

* Logit model 
logit yd tdta

* Classification with cutoff
estat class, cutoff(0.56)
estat class, cutoff(0.7)

* Sensitivity and Specificity
lsens

* ROC Curve
lroc


	** Question 16 **
	*****************

* Logit with a list of preferred variables (3 models)

	* Model 1: correlation set
asdoc logit yd tdta reta opita ebita, save(CreditScoringTables) replace
estat class, cutoff(0.5)
lsens
lroc

rocgold yd tdta reta opita ebita, graph
rocgold yd tdta reta opita ebita


	* Model 2: diversified set
asdoc logit yd tdta reta lsls gempl nwcta mveltd fata, save(CreditScoringTables) replace
estat class, cutoff(0.5)
lsens
lroc

rocgold yd tdta reta opita lsls gempl nwcta mveltd fata, graph
rocgold yd tdta reta opita lsls gempl nwcta mveltd fata

	* Model 3: all variables set
asdoc logit yd $xlist, save(CreditScoringTables) replace
estat class, cutoff(0.5)
lsens
lroc

rocgold yd $xlist, graph
rocgold yd $xlist



	** Question 17 **
	*****************
	
/*************************************************************
INFLUENCE OF OBSERVATIONS OUTLIERS
resdev: deviance residual, residus pondérés
reschi: Pearson residual
difchisq: one step difference in Pearson Chisquare
h: diagonal element of the Hat Matrix "leverage on parameters"
*************************************************************/

	* Model 1: Baseline model
logit yd tdta

* Standardized Pearson residual have already been computed in question 12

* Predicting the tdta distress for model 1
predict plogittdta
label variable plogittdta "logit predicted tdta distress"
graph twoway connected plogittdta tdta || scatter yd tdta || connected lpredtdta tdta

* Outliers of Pearson Standardized Residuals (>|2|)
graph twoway scatter rstandardlog tdta, yline(2) yline(-2) yline(0)
graph twoway scatter rstandardlog plogittdta, yline(2) yline(-2) yline(0)


	
	** Question 18 **
	*****************

* No code needed, answer available in the report

save Chapter1Estimation.dta, replace

* /!\ End of analysis of the estimation sample
* /!\ Next Chapter and questions are dedicated to the validation sample 

****************************************************************************************************
****************************************************************************************************
}


		**********************************
		* CHAPTER II : VALIDATION SAMPLE *
		**********************************
			
{
********** SETUP **********
clear all
set more off
version 14.0
set linesize 90
set scheme s1mono

* install packages
*ssc install asdoc

* Directory 
cd "C:\~\Homework Credit Scoring - Bouillot Janbek Louafi\Code"

* Log capture
capture log close
log using defaut-v1.txt, text replace

* Open data file
* Remark: Data are already sorted by yd tdta 
use "C:\~\Homework Credit Scoring - Bouillot Janbek Louafi\Code\defaut2000bis.dta", clear

* Proc content
des


		******************************
		* Part I : Cleaning the Data *
		******************************

	
	* STEP 1: Labeling our data 
	***************************

label variable yd      "Indic DifficultÈs FinanciËres" 
label variable tdta    "Dette/Actif"   /* total debt/total assets */
label variable reta    "Reserves et report ‡ nouveau/Actif" /* retained earnings */
label variable opita   "Resultat/Actif"  /* operating income/total assets */
label variable ebita   "EBIT ou EBE/Actif" /* earnings before interest taxes/total assets*/

label variable lsls    "Log(ventes)" /* log sales */
label variable lta     "Log(actif)"  /* log assets */
label variable gempl   "Croissance des salariÈs" /* growth of employment */
label variable invsls  "Stocks/Ventes"  /* inventories/sales */
label variable nwcta   "Capital circulant net/Actif" /* net working capital/total assets */

label variable cacl    "Actif circulant/Passif circulant" /* current assets/current liabilities */
label variable qacl    "Disponibilites+VMP/Passif circulant" /* quick assets/current liabilities */ 
label variable fata    "Actifs Fixes/Actif" /* fixed assets/totat assets */
label variable ltdta   "Dette Long Terme/Actif" /* long term debt/total assets */
label variable mveltd  "Valeur de marche Actions/Dette long terme" /* market value equity/lgterm debt */

label define ydl 0 "No Fin.Dis." 1 "Fin.Dist"
label values yd ydl


	* STEP 2: Sort and clean data 
	*****************************

* Sorting data by yd and ebita
sort yd ebita

* Cleaning data (by replacing the -99.98 observations by a "missing obs.")
replace fata = . if fata <= -99.98
replace ltdta = . if ltdta <= -99.98


	* STEP 3: Creating the Validation sample
	****************************************

*list yd $xlist if mod(_n,2) 
*list yd $xlist if !mod(_n,2)

* Creating the VALIDATION sample
drop if mod(_n,2)
describe

* Save Clean dataset for Validation
save CleanDataValidation.dta, replace
clear 

****************************************************************************************************
****************************************************************************************************


		********************************
		* Part II : Questions 19 to 21 *
		********************************

		
* We use the Clean data validation dataset
clear all
u "CleanDataValidation.dta"

	** Question 19 **
	*****************

* Same analysis as in question 15 (but swapping estimation and validation samples)

	* Model 1: Baseline 

* Determining cutoff
asdoc cutpt yd tdta, noadjust save(CreditScoringTables) replace
roctab yd tdta, graph msymbol(none) addplot(scatteri `e(sens)' `=1 - e(spec)') legend(label(3 "Cutpoint"))
bootstrap e(cutpoint), rep(100): cutpt yd tdta, noadjust 

* Logit model 
asdoc logit yd tdta, save(CreditScoringTables) replace

* Classification with cutoff
estat class, cutoff(0.5)

* Sensitivity and Specificity
lsens

* ROC Curve
lroc


	* Model 2: Diversified 

* Logit model 
asdoc logit yd tdta reta lsls gempl nwcta mveltd fata, save(CreditScoringTables) replace

* Classification with cutoff
estat class, cutoff(0.5)
estat class, cutoff(0.7)

* Sensitivity and Specificity
lsens

* ROC Curve
lroc


	** Question 20 **
	*****************

* No code needed, answer available in the report

	
	** Question 21 **
	*****************

/* In this question, we present four of our own original estimations:
	- Model 1 : Principal Components Analysis (PCA)
	- Model 2 : Decision Tree model (TM)
	- Model 3 : Random Forest (RF)
	- Model 4 : Bayesian Networks (BN)

Those models are available in the R code documents "Question 21 - Part I" and "Question 21 - Part II"
	*/

	
	
* We save our dataset for the Part II estimation 
save Chapter2Validation.dta, replace
clear all
}


*****************************************************************************************************************************************************************
* End of Empirical Application - Credit Scoring 
*****************************************************************************************************************************************************************




