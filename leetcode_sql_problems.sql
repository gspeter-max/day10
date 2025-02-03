# Write your MySQL query statement bewi

'''1890. The Latest Login in 2020'''
https://leetcode.com/problems/the-latest-login-in-2020/

with temp_temp as (
    select user_id , 
    max(time_stamp) as time_stamp
    from logins
    where time_stamp >= '2020-01-01 00:00:00' and 
    time_stamp <  '2021-01-01 00:00:00'
    group by user_id 

)
select user_id , 
    time_stamp  as last_stamp
from temp_temp;




'''1907. Count Salary Categories'''
https://leetcode.com/problems/count-salary-categories/

SELECT 'Low Salary' AS category, COUNT(*) AS accounts_count
FROM accounts
WHERE income < 20000

UNION ALL

SELECT 'Average Salary' AS category, COUNT(*) AS accounts_count
FROM accounts
WHERE income >= 20000 AND income <= 50000

UNION ALL

SELECT 'High Salary' AS category, COUNT(*) AS accounts_count
FROM accounts
WHERE income > 50000;



'''1934. Confirmation Rate'''
https://leetcode.com/problems/confirmation-rate/

# Write your MySQL query statement below
with temp_temp as (
    select user_id, 
    round((sum(case when action = 'confirmed' then 1 else 0 end )  / count(*)),2) as confirmation_rate
    from confirmations 
    group by user_id 
)
select s.user_id, 
    coalesce(t.confirmation_rate,0) as confirmation_rate
from signups s 
left join temp_temp t on s.user_id = t.user_id ; 
