

# https://www.cancerimagingarchive.net/collection/rider-lung-ct

https://nbia.cancerimagingarchive.net/nbia-search/?CollectionCriteria=RIDER%20Lung%20CT

RIDER-1129164940


downloaded-manifest.tcia
```
downloadServerUrl=https://nbia.cancerimagingarchive.net/nbia-download/servlet/DownloadServlet
includeAnnotation=true
noOfrRetry=4
databasketId=manifest-1724796815128.tcia
manifestVersion=3.0
ListOfSeriesToDownload=
1.3.6.1.4.1.14519.5.2.1.48822327301563949480198360763011173678
1.3.6.1.4.1.14519.5.2.1.74640583883665924899120840733688469127
```

https://wiki.cancerimagingarchive.net/display/Public/TCIA+REST+API+Guide

```
export MYUID=1.3.6.1.4.1.14519.5.2.1.48822327301563949480198360763011173678
curl https://services.cancerimagingarchive.net/services/v4/TCIA/query/getImage?SeriesInstanceUID=${MYUID} -o image-1.zip
export MYUID=1.3.6.1.4.1.14519.5.2.1.74640583883665924899120840733688469127
curl https://services.cancerimagingarchive.net/services/v4/TCIA/query/getImage?SeriesInstanceUID=${MYUID} -o image-2.zip

mkdir image-1
unzip image-1.zip -d image-1
mkdir image-2
unzip image-2.zip -d image-2

+ then open dicom series in itksnap /3dslicer

+ save as .nii.gz

+ create test masks in lung.nii.gz and landmarks.nii.gz

```

# use itk to create 