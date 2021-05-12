"""
ChainRad
========

File: unit test for the dataset
"""


from os.path import join

import pandas as pd

from core import check_and_get_basics, META_DIR


# pylint: disable=invalid-name
#         The variable is_successful is actually not a constant.


diseases_basics = check_and_get_basics()

failed, passed = 0, 0
for disease, meta_file_prefix in diseases_basics.items():
    patients = []
    print('Testing {}...'.format(disease))
    data = pd.read_csv(join(META_DIR, 'train_{}.csv'.format(meta_file_prefix)),
                       sep='\t')
    print('--- Test: Only one disease per patient in train: ', end='')
    if data['Patient ID'].count() == data['Patient ID'].nunique():
        print('PASSED')
        passed += 1
    else:
        print('FAILED')
        failed += 1
    print('--- Test: Positive and negative cases have same count in train: ',
          end='')
    target_counts = data['target'].value_counts()
    if target_counts[0] == target_counts[1]:
        print('PASSED')
        passed += 1
    else:
        print('FAILED')
        failed += 1
    for patient in data['Patient ID'].values.tolist():
        patients.append(patient)
    data = pd.read_csv(join(META_DIR, 'test_{}.csv'.format(meta_file_prefix)),
                       sep='\t')
    print('--- Test: Only one disease per patient in test: ', end='')
    if data['Patient ID'].count() == data['Patient ID'].nunique():
        print('PASSED')
        passed += 1
    else:
        print('FAILED')
        failed += 1
    is_successful = True
    for patient in data['Patient ID'].values.tolist():
        if patient in patients:
            is_successful = False
        patients.append(patient)
    print('--- Test: No identical patinet(s) in train and test: ', end='')
    if is_successful:
        print('PASSED')
        passed += 1
    else:
        print('FAILED')
        failed += 1
    data = pd.read_csv(join(META_DIR, 'valid_{}.csv'.format(meta_file_prefix)),
                       sep='\t')
    print('--- Test: Only one disease per patient in valid: ', end='')
    if data['Patient ID'].count() == data['Patient ID'].nunique():
        print('PASSED')
        passed += 1
    else:
        print('FAILED')
        failed += 1
    is_successful = True
    for patient in data['Patient ID'].values.tolist():
        if patient in patients:
            is_successful = False
        patients.append(patient)
    print('--- Test: No identical patinet(s) in train, test and valid: ',
          end='')
    if is_successful:
        print('PASSED')
        passed += 1
    else:
        print('FAILED')
        failed += 1
print('Overall test: ', end='')
if failed == 0:
    print('PASSED', end='')
else:
    print('FAILED', end='')
print(' --- passes: {}; fails: {}'.format(passed, failed))
