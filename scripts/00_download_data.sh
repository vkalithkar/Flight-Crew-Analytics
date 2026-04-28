#!/bin/bash

WEBSITE_LOC="data-site.htm"

cd data

# getting the flight crew CSV zip files from nasa data website (html parsing)
grep -Po 'https://[^" >]+\.zip' "$WEBSITE_LOC" | while read line; do 
	wget -c --show-progress \
	--wait=15 \
	--random-wait \
	--timeout=60 \
	-P raw/ "$line"
done

cd raw/ 

# unzipping all of those files
for file in *.zip; do unzip $file; done

# clean up
rm *.zip

# return to root
cd ..
cd ..
