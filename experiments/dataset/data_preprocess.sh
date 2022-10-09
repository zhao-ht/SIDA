# download Office-31 from https://drive.google.com/open?id=0B4IapRTv9pJ1WGZVd1VDMmhwdlE
mv domain_adaptation_images.tar.gz Office-31
cd Office-31
tar -zxvf domain_adaptation_images.tar.gz
cd amazon
cd images
mv * ../
cd ..
rm -r images
cd ..
cd dslr
cd images
mv * ../
cd ..
rm -r images
cd ..
cd webcam
cd images
mv * ../
cd ..
rm -r images
cd ..
cd ..

# download Office-Home from https://drive.google.com/file/d/0B81rNlvomiwed0V1YUxQdC1uOTg/view?usp=sharing&resourcekey=0-2SNWq0CDAuWOBRRBL7ZZsw
mv OfficeHomeDataset_10072016.zip OfficeHomeDataset
cd OfficeHomeDataset
unzip  OfficeHomeDataset_10072016.zip
cd OfficeHomeDataset_10072016
mv * ../
cd ..
rm -r OfficeHomeDataset_10072016
cd ..

# download VisDA2017 from
# https://drive.google.com/file/d/0BwcIeDbwQ0XmdENwQ3R4TUVTMHc/view?usp=sharing
# https://drive.google.com/file/d/0BwcIeDbwQ0XmUEVJRjl4Tkd4bTA/view?usp=sharing
# https://drive.google.com/file/d/0BwcIeDbwQ0XmdGttZ0k2dmJYQ2c/view?usp=sharing
mv train.tar VisDA-2017
mv validation.tar VisDA-2017
mv test.tar VisDA-2017
cd VisDA-2017
tar xvf train.tar
tar xvf validation.tar
tar xvf test.tar
cd ..