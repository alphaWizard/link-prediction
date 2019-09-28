This dataset includes check-ins, tips and tags data of restaurant venues in NYC collected from Foursquare from 24 October 2011 to 20 February 2012.
It contains three files in tsv format, including 3112 users and 3298 venues with 27149 check-ins and 10377 tips.

1. dataset_ubicomp2013_checkins.txt has two columns. 
Each line represents a check-in event. 
The first column is user ID, while the second column is venue ID. 

2. dataset_ubicomp2013_tips.txt has three columns. 
Each line represents a tip/comment a user left on a venue. 
The first and second columns are user ID and venue ID, repsectively. 
The third column is tip text.

3. dataset_ubicomp2013_tags.txt has two columns. 
Each line represents the tags users added to a venue. 
The first column is venue ID while the second column is tag set of the corresponding venues.
Empty tag set may exist for a venue since no user has ever added a tag to it.

=============================================================================================================================
Please note that all user and venue IDs are anonymized, but they can be matched across the three files.
In this dataset, tips may be observed even a user didn't checkin at a venue, because check-ins are collected in a period of four months while tips are collected without time limitation.
Please cite our paper if you publish material based on this dataset.

=============================================================================================================================
REFERENCES
@inproceedings{yang2013fine,
  title={Fine-grained preference-aware location search leveraging crowdsourced digital footprints from LBSNs},
  author={Yang, Dingqi and Zhang, Daqing and Yu, Zhiyong and Yu, Zhiwen},
  booktitle={Proceedings of the 2013 ACM international joint conference on Pervasive and ubiquitous computing},
  pages={479--488},
  year={2013},
  organization={ACM}
}

@inproceedings{yang2013sentiment,
  title={A sentiment-enhanced personalized location recommendation system},
  author={Yang, Dingqi and Zhang, Daqing and Yu, Zhiyong and Wang, Zhu},
  booktitle={Proceedings of the 24th ACM Conference on Hypertext and Social Media},
  pages={119--128},
  year={2013},
  organization={ACM}
}

