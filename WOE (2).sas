%let variable = Exited; 
%let varlist = CreditScore, Age, Tenure, Balance, NumOfProducts;
%let final = Fr_%scan(&varlist., 1);

/* Macro to check if dataset exists and delete if it does */
%macro checkds(final);
    %if %sysfunc(exist(&final)) %then %do;
        proc delete data=&final; 
        run;
    %end;
    %else %do;
        data _null_;
            file print;
            put #3 @10 "Data set &final. does not exist";
        run;
    %end;
%mend checkds;

/* Check and delete the dataset if it exists */
%checkds(final_result_%scan(&varlist., 1));

options mlogic symbolgen mprint;

/* Macro to calculate Weight of Evidence (WoE) */
%macro woe();
    /* Loop through each variable in varlist */
    %do i = 1 %to %sysfunc(countw(&varlist.));
        /* Step 1: Rank data into deciles */
        proc rank data=my.logistic_ds2_train groups=10 out=step7;
            var %scan(&varlist., &i.);
        run;

        /* Step 2: Calculate Total lapsed and non-lapsed counts */
        proc transreg data=step7;
            model identity(&variable.) = monotone(%scan(&varlist., &i.));
        run;

        proc sql;
            create table step8 as
            select *, 
                   sum(&variable.=1) as Total_lapsed,
                   sum(&variable.=0) as Total_non_lapsed
            from step7
            group by %scan(&varlist., &i.);
        quit;

        /* Step 3: Calculate WoE and IV for each variable */
        proc sql;
            create table table_&i. as
            select %scan(&varlist., &i.),
                   sum(&variable.=0) as non_lapsed,
                   sum(&variable.=1) as lapsed,
                   log((sum(&variable.=1)/sum(&variable.=0)) / 
                       (mean(Total_lapsed) / mean(Total_non_lapsed))) as woe,
                   (sum(&variable.=1)/mean(Total_lapsed) - 
                    sum(&variable.=0)/mean(Total_non_lapsed)) * calculated woe as iv
            from (select &variable., %scan(&varlist., &i.), Total_lapsed, Total_non_lapsed
                  from step8)
            group by %scan(&varlist., &i.);
        quit;

        /* Step 4: Summarize IV for each variable */
        proc sql;
            create table fr_%scan(&varlist., &i.) as 
            select "%scan(&varlist., &i.)" length=40, 
                   sum(iv) as sum_iv 
            from table_&i.;
        quit;

        /* Append results to the first variable's results */
        proc append base=fr_%scan(&varlist., 1) data=fr_%scan(&varlist., &i.) force; 
        run;
    %end;
%mend;

/* Execute the WoE macro */
%woe();

/* Final dataset creation with Comments based on IV values */
data final1;
    set fr_%scan(&varlist., 1);
    length Comments $15;

    /* Assign comments based on sum_iv thresholds */
    if sum_iv <= 0.02 then Comments = "Not Rel";
    else if sum_iv > 0.02 and sum_iv <= 0.1 then Comments = "Weak";
    else if sum_iv > 0.1 and sum_iv <= 0.3 then Comments = "Medium";
    else if sum_iv > 0.3 and sum_iv <= 0.5 then Comments = "Strong";
    else Comments = "Not Usable";
run;