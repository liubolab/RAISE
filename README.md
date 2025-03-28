# README

'demo.py' is a model calculation process obtained from 'demo.csv' data.

## Data format specification

<table>
	<tr>
	    	<th>Catergory</th>
	    	<th colspan="5">Demographic Information</th>
		<th colspan="7">Pregnancy Event</th>
	</tr>
	<tr >
	    	<td>Behavior</td>
	    	<td>Current year</td>
	    	<td>Birth year</td>
		<td>Ethnicity</td>
		<td>Education</td>
		<td>Hukou</td>
		<td>Pregnancy year</td>
		<td>Pregnancy month</td>
		<td>Delivery mode</td>
		<td>Pregnancy outcome</td>
		<td>Survival status of children</td>
		<td>Next pregnancy year</td>
		<td>Next pregnancy month</td>
	</tr>
	<tr>
	    	<td>Format</td>
	    	<td>yyyy</td>
		<td>yyyy</td>
		<td>mapped label</td>
		<td>mapped label</td>
		<td>mapped label</td>
		<td>yyyy</td>
		<td>m/mm</td>
		<td>mapped label</td>
		<td>mapped label</td>
		<td>mapped label</td>
		<td>yyyy</td>
		<td>m/mm</td>
	</tr>
</table>

- See 'demo.csv' for an example
- Only fill in the numbers corresponding pregnancy behaviors and remove the headers of the file when the file is uploaded
- Demographic Information is required to be provided only once.
- Please complete the corresponding Pregnancy Event behavior based on the specific details of each pregnancy. After filling out one complete pregnancy event, proceed to document subsequent pregnancies until all previously experienced pregnancies have been recorded.
- For the last pregnancy, next pregnancy year/month is when you expect your next pregnancy to be.

## Web functionality

### Data upload

- Method 1: Directly download the data format template, fill it out according to the specified requirements, and upload the completed CSV file
  - The template is named `mod.csv`
  - Refer to the "Data format specification" section for requirements.
- Method 2: Manual entry
  - Single input: Current year, Birth year, Ethnicity, Education, Hukou
  - Iteratively input each pregnancy event in its entirety until no new data remains:
    - Fill in the 7 behaviors for each pregnancy event: Pregnancy month, Pregnancy year, Delivery mode, Pregnancy outcome, Survival status of children, Next pregnancy year, Next pregnancy month.

### Display output

The probability of the four pregnancy outcomes

## Front-end design

Refer to the `front end.pptx` file.
