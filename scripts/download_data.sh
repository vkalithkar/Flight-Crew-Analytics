#!/bin/bash

WEBSITE_LOC="data-site.htm"

cd data

# trying to get all of the zip files from nasa data website (html parsing)
#for line in $(grep -Po 'https://[^" >]+\.zip' $WEBSITE_LOC); do wget -P raw/ $line;>
#for line in $(grep -Po 'https://[^" >]+\.zip' $WEBSITE_LOC); do echo $line;done

grep -Po 'https://[^" >]+\.zip' "$WEBSITE_LOC" | wget -c --show-progress -P raw/ -i -


cd raw/ 

# unzipping all of those files
for file in *.zip; do unzip $file; done

rm *.zip

cd ..
cd ..
