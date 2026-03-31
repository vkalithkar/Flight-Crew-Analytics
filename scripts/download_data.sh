#!/bin/bash

WEBSITE_LOC="data-site.htm"

cd data

for line in $(grep -Po 'https://[^" >]+\.zip' $WEBSITE_LOC); do wget -P raw/ $line; done

cd raw/ 

for file in *.zip; unzip $file; done

rm *.zip

cd ..
cd ..
