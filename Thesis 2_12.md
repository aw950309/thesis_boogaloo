# Predicting High-Risk Wildlife Zones in Sweden with Machine Learning and Environmental Data

# 1.0 Introduktion

Sweden faces significant challenges related to human-wildlife interactions. According to Viltvårdsverket a vehicle collision with wild animals occurs every 7th minute in Sweden, with over 75.000 accidents happening yearly. Wildlife-vehicle collisions (WVCs), particularly with moose (alces alces), but also including roe deer (*Capreolus capreolus*) and wild boars (sus scrofa), represent a serious road safety issue leading to personal injuries, significant economic costs and in rare occasions even fatalities with an average of 5 people dying every year. (Source: data from Trafikverket/National Wildlife Accident Council). 

## **1.1 Problem statement**

Current preventive measures (e.g., static warning signs, fences, control of animal population) are often reactive and lack the dynamism to adapt to the complex movement patterns of animals, which are influenced by season, weather, and landscape changes.. Earlier models relied primarily on logistic regression. While effective for identifying linear relationships, these models may fail to capture complex non-linear interactions between environmental variables and collision risk. 

Finally, the wildlife landscape in Sweden has changed; while moose have historically been the focus since their collisions have led to the most amount of fatalities, wild boar populations have increased rapidly since 2005, and roe deer along with the European fallow deer constitute the vast majority of collisions, necessitating an analysis that extends beyond a single species. There is a need to move beyond linear, single species predictions  and to evaluate which factors dynamically drive collision risks across different wildlife types.

## **1.2 Research objective**

We will therefore evaluate whether adding temporally resolved environmental predictors (including weather/climate proxies) and non-linear modellings improves predictive performance, following established practices in occurrence-based predictive mapping, while being motivated by evidence of seasonally varying animal behavioral responses to roads (Beyer et al., 2013\) and not just simply those of moose like previous models.

This thesis aims to apply this approach to a Swedish context. Using Python and a Design Science approach, we will design, develop, and evaluate  a data-driven prediction modelling artifact system (including machine learning and statistical models) that can predict high-risk zones for a specific wildlife management problem, using frequently occurring wildlife-vehicle collisions as a case study. We aim to address the following research question: Which environmental factors and temporal variables are the most significant determinants of animal-vehicle collision risk in Sweden when analysing WVC data through non-linear modelling? By integrating accident statistics with temporal and environmental variables along with types of animal, we aim to identify the variables that contribute most significantly to high risk zones for WVC:s. 

## **1.3 Research question and aim**

Main-

Which environmental factors and temporal variables are the most significant predictors for \[for different species\] of animal-vehicle collision risk in Sweden when analysing WVC data through-non linear modelling?

Sub- 

SUB1. Which environmental and road-traffic variables are most strongly associated with WVC location in our Swedish dataset? 

How do temporal factors (season, time of day, month, hunting season) influence WVC patterns across different animal species in Sweden?

How does the identification of these non-linear determinants improve predictive accuracy  
compared to a logistic-r traditional models \[Seiler, neural network vs linear regression\]?

The purpose of this is to get a greater understanding on what affects animals to get hit by cars and by knowing this be able to further educate drivers and cars on what to look out for.

**1.3 Scope and delimitations**

Only data from 2015 and onwards

# 2 Extended background 

## **2.1 International approaches to wildlife predictive modelling**

Data and systems science has made significant advancements internationally in predictive modeling for wildlife management. Methodologies that link spatial records to environmental predictors have proven effective for monitoring efforts at large scales. A prominent example is the management of invasive Feral Hogs in the USA (McClure et al., 2015\).There, observation data is combined with environmental variables (e.g., vegetation, proximity to water, temperature) to build predictive models using machine learning to determine where the pigs are likely to be found. This methodology has proven effective for prioritising monitoring and intervention efforts.

## **2.2 The Swedish context: Existing models and limitations**

A closely related predictive modelling tradition also exists for Swedish WVC:s  Seiler modelled and predicted locations of moose–vehicle collisions in Sweden, demonstrating that collision risk can be spatially estimated from landscape and infrastructure correlates (Seiler, 2005). However, as noted in the problem statement, significant gaps remain in the existing predictive traditions for Swedish WVC:s. Earlier models such as Seiler’s, relied primarily on logistic regression. While effective for identifying linear relationships, these models may fail to capture complex non-linear interactions between environmental variables and collision risk. 

## **2.3 Ny titel: From parametric to non-parametric modeling (gammal Machine learning vs statistical modeling)**

The distinction between traditional statistical modelling and machine learning approaches represents a methodological divide in predictive analytics. Classical statistical methods, such as logistic regression, attempt to model relationships through predetermined functional forms, typically assuming linear relationships between predictors and outcomes (Berk, 2020). In contrast, machine learning approaches like Random Forests operate without such parametric assumptions, allowing the data itself to reveal complex patterns (Berk, 2020).

### 2.3.1 The Linear Constraint and Non-Linear Reality

Random Forests, introduced by Breiman (2001), construct predictions through an ensemble of decision trees, each capable of capturing non-linear relationships and high-order interactions without explicit specification. This architecture enables the detection of threshold effects—for instance, recognising that temperature decreases affect wildlife behaviour differently above versus below freezing—which linear models cannot inherently capture (Berk, 2020). Cutler et al. (2007) demonstrated that for ecological data characterised by such complex interactions, Random Forests consistently outperform traditional regression approaches (Berk, 2020).

### 2.3.2 Feature Importance

A significant advantage of ensemble methods is their ability to quantify variable importance, revealing which predictors most strongly influence outcomes (Berk, 2020). This permits researchers to identify whether factors like snow depth or speed limits dominate collision risk without the need to pre-specify interaction terms.

## **2.4 Spatiotemporal factors and environmental  predictors**

The spatial distribution of WVC:s is rarely uniform. Road-mitigation planning highlights that collision hotspots often do not align with natural animal movement corridors, as risk is heavily influenced by specific road characteristics and traffic patterns (Lee et al., 2020). This mismatch necessitates a rigorous variable selection process to ensure that predictive artefacts provide safety-oriented decision support rather than simple ecological mapping.

Furthermore, the temporal dimension introduces significant complexity into these spatial risks. Ungulate behavioural responses to road density are known to vary by season, suggesting that temporally structured predictors are critical for a Swedish setting (Beyer et al., 2013). This seasonal variation is largely driven by vegetation phenology, the timing of plant growth, this is where the spring "green-up" attracts animals to nutrient-rich forage in roadside verges. Conversely, winter snow depth acts as a physical constraint, frequently "funnelling" animals onto cleared road surfaces to avoid the energetic cost of moving through deep snow.

By integrating these spatiotemporal variables, this research moves beyond static risk mapping. It acknowledges that a "high-risk zone" is not merely a fixed coordinate on a map, but a dynamic state that fluctuates based on meteorological conditions, biological cycles, and infrastructure characteristics. This complexity further justifies the transition from linear statistics to the non-parametric modelling approach (discussed in Section 2.3) capable of capturing these multi-dimensional interactions.

## **2.5 Species specific risk profiles**

Moose, Fallow Deer, Roe Deer, and the boar constituted 91% of accidents in 2025\. As prey animals, these ungulates instinctively rely on a common defense strategy known in the animal kingdom as the freeze response, causing them to remain motionless when confronted with sudden stress or perceived danger. While this strategy may be effective against natural predators by reducing the likelihood of detection, it becomes maladaptive when the threat is an approaching vehicle traveling at high speed. This combined with their high population compared to carnivora results in a

Research demonstrates that habitat suitability models for ungulate roadkill prognosis must account for species-specific movement patterns—red deer, roe deer, and wild boar each require separate pathway simulations (Wierzchowski et al., 2019). Similarly, studies in British Columbia found significant spatiotemporal differences in roadkill patterns between species including water deer and wild boar (Rea et al., 2014\) There is therefore a need to investigate each of them and have them as the focus. The following subsections outline the behavioral and ecological characteristics of each species that justify the variables selected for our model, including hunting season, rutting season, meteorological season, and time of day.

## 2.5.1 The Moose (Alces alces): High Impact and Landscape Correlates

Moose have historically dominated Swedish WVC research because their large body size (300-700 kg) and tall, spindly legs results in the most severe consequences for human safety. Upon impact, the car’s crumple zones hit the legs, causing the heavy upper body to collapse through the windshield or the roof of the car, which can crush the occupants. Their dark coat also blends in with the dark, and their eyes are notoriously difficult to see at night due to the beam of most passenger vehicle headlights not being able to reach their tall stature. Because of this danger of preventative measures are with moose collisions in mind. Seiler's foundational work demonstrated that collision risk can be predicted from landscape and infrastructure variables, with traffic volume, vehicle speed, and fencing being dominant factors (Wierzchowski et al., 2019). Within road categories, forest cover, edge density, and moose abundance significantly distinguished accident from non-accident sites (Wierzchowski et al., 2019).

## 2.5.2 Roe Deer (Capreolus capreolus): High Frequency and Edge Dependency

While moose collisions receive attention due to severity, roe deer account for the vast majority of wildlife-vehicle collisions across Europe. Studies confirm that roe deer are the most common roadkill species among ungulates, with temporal patterns showing clear seasonal and daily variation (Pokorny et al., 2022; Balčiauskas et al., 2020). This high collision frequency makes them essential for any comprehensive WVC prediction model.

Roe deer are crepuscular animals, most active during dawn and dusk periods, which directly justifies my inclusion of time-of-day classification in the model. Research on temporal patterns found that roe deer collisions peak during specific hours corresponding to twilight activity Kim & Lee (2023). Their preference for ecotone habitats—transition zones between woodland and open fields—differs from deep-forest species and suggests that landscape variables capturing forest-agriculture boundaries will be important predictors (Seiler, 2005).

The roe deer rutting season occurs in July-August, earlier than moose, creating a distinct temporal risk window that supports including species-specific rutting periods. Additionally, collision patterns show strong monthly variation, with peaks often occurring in spring during dispersal and autumn during increased movement (Pokorny et al., 2022; , Balčiauskas et al., 2023). These patterns justify include meteorological season as a variable.

## 2.5.3 Wild Boar (Sus scrofa): The Growing Nocturnal Threat

Wild boar populations have increased dramatically across Europe, fundamentally changing the wildlife-vehicle collision landscape. In Sweden, this expansion since 2005 means older predictive models may not capture current collision risks. Studies confirm that wild boar collision patterns differ substantially from deer species in both spatial distribution and temporal occurrence (Seiler, 2005, Kim & Lee (2023).

Unlike moose and deer which are most active at twilight, wild boar exhibit primarily nocturnal behavior. Research on temporal patterns found that wild boar collisions peak during nighttime hours, with seasonal peaks occurring in autumn (October) rather than following the same patterns as deer Balčiauskas et al., 2023), Vrkljan et al. (2020). This behavioral difference necessitates temporal variables distinguishing night from other periods, not just dawn and dusk.

Wild boar also travel in family groups called sounders, meaning multiple animals may cross roads in succession. Their habitat preferences include forests with mast availability and agricultural areas, potentially creating different landscape correlates than deer species (Seiler, 2005; . The combination of nocturnal activity, recent population expansion, and distinct seasonal patterns (including autumn breeding activity) makes wild boar a key driver for robust temporal variables in my analysis.

## 2.5.4 Fallow Deer (Dama dama): Localized High-Density Populations

European fallow deer present an interesting case in Swedish WVC patterns. Despite having a smaller total population than moose, they account for substantial collision numbers due to their distinctive distribution pattern. Unlike the widespread moose or ubiquitous roe deer, fallow deer occur in localized high-density populations, often associated with historical introductions to estates from which they have spread (Seiler, 2005).

Similar to roe deer, fallow deer show highest densities in ecotone habitats at transitions between woodland and open fields (Dussault et al., 2007). They also exhibit crepuscular activity patterns, though research suggests some variation in peak activity times compared to roe deer (Balčiauskas et al., 2020). The autumn rutting season (October-November) involves males establishing display territories, increasing movement and potentially reducing vigilance.

This concentrated distribution means collision risk is highly location-specific—areas with fallow deer populations experience disproportionately high rates while nearby areas have none. For my model, this suggests spatial clustering may be particularly pronounced for fallow deer, and that population density information could be especially valuable. Their localized distribution also means that variables effective for widespread species may perform differently for fallow deer predictions.

## 2.5.5 Comparative summary

The preceding sections reveal distinct temporal risk profiles across species. Roe deer and fallow deer show crepuscular patterns with collisions peaking at dawn and dusk, while wild boar collisions occur predominantly at night (Dussault et al., 2007). Seasonally, roe deer collisions peak in spring-summer corresponding to their July-August rut, whereas moose risk increases during the autumn rut (September-October), and wild boar show autumn peaks in October (Dussault et al., 2007). These differences justify analyzing species separately and including both time-of-day and season as key model variables. The variation also suggests that a single predictive model may perform differently across species, supporting our multi-species analytical approach.

## **2.6 The model as a Decision Support Artifact** 

Within the framework of Design Science Research (DSR), an artifact is defined as an object created by humans with the intention of addressing a practical problem (Johannesson & Perjons, 2014, p. 3). The predictive model developed in this thesis functions as such an artifact. Specifically acting as a model and instantiation that transforms environmental data into prescriptive/actionable insights that support prescriptive decision-making for risk assessment (Johannesson & Perjons, 2014, pp. 11, 29). DSR emphasizes that such artifacts must not only solve local practical problems but also demonstrate novelty and contribute knowledge of general interest (Johannesson & Perjons, 2014, pp. 8).

The model’s primary contribution lies in its role as a foundational "proof of concept." By demonstrating the artifact’s feasibility in a specific case, this thesis validates the artifact's potential utility (Johannesson & Perjons, 2014, p. 133). Rather than prescribing specific interventions, the artifact provides a unified prediction framework that contributes to a global practice, allowing future researchers or governmental bodies like Trafikverket to extend and adapt the solution to new contexts (Johannesson & Perjons, 2014, p. 11). The modular nature of the applied machine learning approaches enables the incorporation of additional variables or alternative algorithms. By establishing which environmental factors serve as significant predictors of wildlife–vehicle collision risk, this artifact creates a rigorous empirical foundation (Johannesson & Perjons, 2014, p. 5\) upon which sophisticated decision support systems can be constructed.

# 3 Method

3.1 Research design

This study follows a Design Science Research approach combined with quantitative data analysis and machine learning techniques. In line with design science research principles (Johannesson & Perjons, 2014), the primary objective is to design, develop and evaluate a predictive modeling artifact capable of identifying high-risk wildlife-vehicle collisions zones in Sweden. 

The research process follows five main steps: 

1. Data preparation  
2. Data preprocessing and variable engineering  
3. Exploratory analysis  
4. Model development (non-linear and linear models)  
5. Model comparison and interpretation of results.

All analyses are conducted using Python within a Jupyter Notebook environment. The workflow will rely on established data science libraries such as Pandas and NumPy.

# 3.2 Data collection

The analysis will be combining multiple datasets:

3.2.1 Wildlife-Vehicle collision data  
Collision records from 2015 onwards are obtained from Nationella viltolycksrådet (NVR). Each records include: 

* Location (geographic coordinates)  
* Date and time  
* Species involved

Only data from 2015 onward are included because before that it was measured differently lol

## 

## 3.2.2 Meteorological data

Weather data are spatially and temporally matched to each collision record. Variables include:

* Temperatures  
* Precipitation (Rain/snow)  
* Seasonal classification


## 3.2.3 Road and infrastructure data

Road related predictors include:

* Speed limit  
* Presence of wildlife fencing


These variables are included based on previous findings showing strong associations between infrastructure characteristics and collision risk (Seiler, 2005\)

## 3.2.4 Wildlife and management variables

Species specific and management related variables include:  
	Animal species (moose, roe deer, wild boar, fallow deer)  
	Population density  
	Hunting season  
	Rutting season  
	Projected increase or decrease in species

# 3.3 Variable preparation

Several variables are prepared before modeling.  
	Time of day is classified as day, night, dawn or dusk  
	Temperatures are split into Very cold, cold, 0, warm and very warm :3.  
	Hunting and rutting seasons are coded as binary (1=yes, 0=no)

# 

# 3.4 Model development

To answer the research question, both linear and non-linear models are used.

The linear model will serve as a comparison for more advanced models.

To capture non linear relationships and interactions between variables, tree-based machine learning models are applied.

