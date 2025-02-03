'''
Problem 1: Time-Weighted Moving Average (Stock Data Analysis)

Each stocks price is recorded at irregular timestamps. 
You need to compute a time-weighted moving average over the past 1 hour for each recorded timestamp of every stock. 
'''

with temp_temp as (
	select a.stock_id,  
		a.recorded_at , 
		avg(a.price  * (timestampdiff(second , b.recorded_at, a.recorded_at)/ 60)) as avg_moving , 
		sum(timestampdiff(second , b.recorded_at , a.recorded_at) / 60) as total_sum 
		from stock_price a
		left join stock_price b on a.stock_id = b.stock_id 
		and b.recorded_at between date_sub(a.recorded_at, interval 1 hour) and a.recorded_at 
		group by a.stock_id, a.recorded_at 
		) 

select stock_id, 
	recorded_at , 
	(avg_moving / coalesce(total_sum, 1)) as twma 

from temp_temp ;


''' 

Problem 2: Identifying the Most Profitable Customer Sequences 

Your task is to find the longest consecutive sequence of purchases by each customer where:

Each transaction is on a consecutive day.
The total amount spent in this sequence is maximum. 
'''


with temp_temp as (
	select  customer_id , 
	transaction_id , 
	transaction_date, 
	amount_spend, 
	date_diff(transaction_date , lag(transaction_date) over(partition by customer_id order by transaction_date )) as diff 
	from transactions_table 	
),
group_data as (
	select * , 
		sum(case when diff is null or diff > 1 then 1 else 0 end) over(partition  by customer_id order by transaction_date ) as group_id 
	from temp_temp 

), 
compute_data as (
	select customer_id , 
	min(transaction_date) as start_date, 
	max(transaction_date ) as end_date, 
	sum(amount_spend) as total_amount_spend

	from group_data
	group by customer_id, group_dataid 

)
select customer_id , 
	start_date , 
	end_date, 
	total_amount_spend
from compute_data 
order by customer_id,
	start_date, 
	end_date, 
	total_amount_spend desc ;  
