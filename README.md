# acs-occ-write-ins
Overview, code, and results from the Industry and Occupation Autocoder project as part of the Census data science program in Spring/Summer 2021. Goal was to create a machine learning (ML) model to predict Census industry and occupation codes to create efficiencies. This part of the project deals exclusively with occupation codes. 

Time and computing constraints limited my ability to train a ML model on the full set of data available to us. However, I demonstrate an "Index Autocoder" that requires no additional computing resources or probabilistic estimates to generate model predictions on the full sample (see the notebook `exact_match_results.ipynb` for details). I achieved such high prediction rates (with presumably 100 percent accuracy provided that the "crosswalk" file is correct) that I decied to implement as a first-stage routine in an overall ensemble pipeline.

The other notebook in this repository (`acs_workflow.ipynb`) implements a full occupation autocoder pipeline on a 3 percent subsample of the data. Note that the input data is considered Title 13 protected. As such, only the jupyter notebooks are presented here. All data presented in these notebooks were reviewed for disclosure avoidance (see below).

The U.S. Census Bureau reviewed this data product for unauthorized disclosure of confidential information and approved the disclosure avoidance practices applied to this release. CBDRB-FY21-POP001-0171.

