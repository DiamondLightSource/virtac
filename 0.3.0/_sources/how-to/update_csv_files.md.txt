# Update CSV Files

Use the CLI to recreate the csv files. This runs create_csv.py and reads data from PVs on port 5064, so ensure you have configured your CA ports to 50XX.

You can run create_csv with the default arguments and the correct ringmode, eg:

:::{code-block} bash
create_csv I04
:::

```{program-output} create_csv -h
```
