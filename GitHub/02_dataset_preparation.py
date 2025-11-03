import os
import shutil
import glob

def main():
    print("Starting dataset preparation script")

    # 1. Configuration
    base_img_dir = './Analysis/images'
    raw_cc_dir = os.path.join(base_img_dir, '01 tongue CC all')
    raw_mask_dir = os.path.join(base_img_dir, '02 tongue mask all')
    clean_cc_dir = os.path.join(base_img_dir, '03 tongue CC')
    clean_mask_dir = os.path.join(base_img_dir, '04 tongue mask')
    train_cc_dir = os.path.join(base_img_dir, '05 train CC')
    train_mask_dir = os.path.join(base_img_dir, '06 train mask')
    test_cc_dir = os.path.join(base_img_dir, '07 test CC')
    test_mask_dir = os.path.join(base_img_dir, '08 test mask')
    all_dirs = [
        clean_cc_dir, clean_mask_dir,
        train_cc_dir, train_mask_dir, test_cc_dir, test_mask_dir
    ]

    # 2. Prepare and clean directories
    def prepare_directories(dir_list):
        print("Preparing and cleaning directories")
        for path in dir_list:
            if os.path.exists(path):
                for item in glob.glob(os.path.join(path, '*')):
                    if os.path.isdir(item):
                        shutil.rmtree(item)
                    else:
                        os.remove(item)
            else:
                os.makedirs(path)
        print("All directories are ready.")

    # 3. Copy and Rename First-Shot Images
    def process_first_shot_images():
        print("\nStep 1: Processing first-shot images")
        for src_file_path in glob.glob(os.path.join(raw_cc_dir, '*_01.png')):
            filename = os.path.basename(src_file_path)
            new_filename = filename.replace('_01.png', '.png')
            dest_file_path = os.path.join(clean_cc_dir, new_filename)
            shutil.copy(src_file_path, dest_file_path)
        for src_file_path in glob.glob(os.path.join(raw_mask_dir, '*_01.png')):
            filename = os.path.basename(src_file_path)
            new_filename = filename.replace('_01.png', '.png')
            dest_file_path = os.path.join(clean_mask_dir, new_filename)
            shutil.copy(src_file_path, dest_file_path)
        print("Finished copying and renaming first-shot images.")

    # 4. Split Data into Training and Testing Sets
    trainset_filenames = ['second_R03.png', 'second_R35.png', 'first_R008.png', 'first_R026.png', 'first_R086.png', 'first_R064.png', 'second_R29.png', 'first_R023.png', 'first_R132.png', 'second_R37.png', 'first_R050.png', 'first_R113.png', 'first_R133.png',
                          'second_R56.png', 'second_R48.png', 'first_R097.png', 'first_R044.png', 'first_R121.png', 'first_R100.png', 'first_R079.png', 'second_R49.png', 'first_R058.png', 'first_R025.png', 'first_R061.png', 'first_R006.png', 'first_R127.png',
                          'second_R32.png', 'first_R011.png', 'first_R095.png', 'first_R088.png', 'second_R40.png', 'first_R106.png', 'second_R43.png', 'second_R53.png', 'second_R10.png', 'first_R066.png', 'first_R082.png', 'first_R116.png', 'first_R063.png',
                          'second_R25.png', 'first_R099.png', 'first_R045.png', 'first_R035.png', 'first_R117.png', 'second_R55.png', 'second_R34.png', 'first_R021.png', 'second_R51.png', 'first_R019.png', 'second_R16.png', 'second_R08.png', 'second_R05.png',
                          'second_R12.png', 'first_R123.png', 'second_R36.png', 'first_R102.png', 'second_R60.png', 'first_R012.png', 'first_R070.png', 'first_R091.png', 'second_R23.png', 'first_R073.png', 'second_R46.png', 'first_R009.png', 'first_R062.png',
                          'first_R054.png', 'first_R053.png', 'first_R034.png', 'first_R047.png', 'first_R119.png', 'first_R087.png', 'first_R112.png', 'second_R44.png', 'first_R125.png', 'first_R033.png', 'first_R118.png', 'first_R101.png', 'first_R098.png',
                          'first_R114.png', 'first_R094.png', 'first_R104.png', 'second_R13.png', 'first_R048.png', 'first_R037.png', 'first_R055.png', 'first_R052.png', 'first_R065.png', 'first_R137.png', 'second_R14.png', 'second_R41.png', 'first_R018.png',
                          'first_R028.png', 'first_R004.png', 'first_R085.png', 'second_R31.png', 'second_R27.png', 'second_R17.png', 'first_R103.png', 'first_R060.png', 'first_R046.png', 'second_R39.png', 'second_R26.png', 'first_R131.png', 'first_R043.png',
                          'first_R074.png', 'second_R02.png', 'first_R020.png', 'second_R59.png', 'first_R136.png', 'first_R077.png', 'first_R078.png', 'first_R130.png', 'first_R124.png', 'first_R067.png', 'second_R04.png', 'first_R109.png', 'first_R135.png',
                          'first_R057.png', 'first_R128.png', 'first_R051.png', 'second_R19.png', 'first_R107.png', 'first_R126.png', 'first_R041.png', 'first_R122.png', 'first_R003.png', 'first_R032.png', 'first_R015.png', 'second_R21.png', 'first_R076.png',
                          'first_R071.png', 'first_R049.png', 'first_R010.png', 'first_R027.png', 'first_R005.png', 'first_R013.png', 'second_R50.png', 'first_R105.png', 'first_R096.png', 'second_R01.png', 'first_R001.png', 'first_R129.png', 'first_R138.png',
                          'second_R52.png', 'first_R081.png', 'first_R031.png', 'second_R42.png', 'first_R108.png', 'first_R075.png', 'second_R09.png', 'first_R038.png', 'first_R059.png', 'second_R30.png', 'first_R068.png', 'second_R22.png', 'first_R115.png',
                          'second_R20.png', 'first_R092.png']
    testset_filenames = ['second_R24.png', 'first_R022.png', 'first_R069.png', 'second_R54.png', 'first_R040.png', 'first_R093.png', 'first_R039.png', 'second_R07.png', 'first_R016.png', 'first_R072.png', 'first_R084.png', 'first_R036.png', 'second_R18.png',
                         'first_R042.png', 'first_R017.png', 'first_R120.png', 'first_R083.png', 'first_R002.png', 'second_R15.png', 'first_R007.png', 'second_R33.png', 'first_R111.png', 'first_R134.png', 'second_R47.png', 'first_R030.png', 'second_R28.png',
                         'first_R090.png', 'first_R014.png', 'second_R06.png', 'first_R110.png', 'second_R58.png', 'first_R029.png', 'second_R45.png', 'first_R056.png', 'second_R57.png', 'first_R080.png', 'first_R024.png', 'second_R38.png', 'first_R089.png',
                         'second_R11.png']

    def split_data():
        print("\nStep 2: Splitting data into training and testing sets")
        for filename in trainset_filenames:
            shutil.copy(os.path.join(clean_cc_dir, filename), os.path.join(train_cc_dir, filename))
            shutil.copy(os.path.join(clean_mask_dir, filename), os.path.join(train_mask_dir, filename))
        for filename in testset_filenames:
            shutil.copy(os.path.join(clean_cc_dir, filename), os.path.join(test_cc_dir, filename))
            shutil.copy(os.path.join(clean_mask_dir, filename), os.path.join(test_mask_dir, filename))
        print("Finished splitting data.")

    # 5. Organize Files into Category Subdirectories
    train_category_list = [['second_R03.png', 0], ['second_R35.png', 1], ['first_R008.png', 2], ['first_R026.png', 2], ['first_R086.png', 1], ['first_R064.png', 0], ['second_R29.png', 2], ['first_R023.png', 0], ['first_R132.png', 2], ['second_R37.png', 1],
                           ['first_R050.png', 0], ['first_R113.png', 0], ['first_R133.png', 0], ['second_R56.png', 1], ['second_R48.png', 1], ['first_R097.png', 2], ['first_R044.png', 1], ['first_R121.png', 0], ['first_R100.png', 1], ['first_R079.png', 1],
                           ['second_R49.png', 2], ['first_R058.png', 0], ['first_R025.png', 1], ['first_R061.png', 2], ['first_R006.png', 1], ['first_R127.png', 1], ['second_R32.png', 1], ['first_R011.png', 0], ['first_R095.png', 1], ['first_R088.png', 0],
                           ['second_R40.png', 1], ['first_R106.png', 2], ['second_R43.png', 0], ['second_R53.png', 2], ['second_R10.png', 2], ['first_R066.png', 1], ['first_R082.png', 1], ['first_R116.png', 1], ['first_R063.png', 0], ['second_R25.png', 2],
                           ['first_R099.png', 1], ['first_R045.png', 0], ['first_R035.png', 1], ['first_R117.png', 1], ['second_R55.png', 2], ['second_R34.png', 2], ['first_R021.png', 1], ['second_R51.png', 0], ['first_R019.png', 1], ['second_R16.png', 2],
                           ['second_R08.png', 1], ['second_R05.png', 2], ['second_R12.png', 2], ['first_R123.png', 1], ['second_R36.png', 0], ['first_R102.png', 2], ['second_R60.png', 2], ['first_R012.png', 0], ['first_R070.png', 0], ['first_R091.png', 1],
                           ['second_R23.png', 1], ['first_R073.png', 1], ['second_R46.png', 0], ['first_R009.png', 1], ['first_R062.png', 1], ['first_R054.png', 1], ['first_R053.png', 0], ['first_R034.png', 1], ['first_R047.png', 1], ['first_R119.png', 0],
                           ['first_R087.png', 1], ['first_R112.png', 1], ['second_R44.png', 1], ['first_R125.png', 1], ['first_R033.png', 1], ['first_R118.png', 2], ['first_R101.png', 1], ['first_R098.png', 0], ['first_R114.png', 0], ['first_R094.png', 1],
                           ['first_R104.png', 1], ['second_R13.png', 0], ['first_R048.png', 1], ['first_R037.png', 0], ['first_R055.png', 2], ['first_R052.png', 1], ['first_R065.png', 1], ['first_R137.png', 0], ['second_R14.png', 2], ['second_R41.png', 0],
                           ['first_R018.png', 0], ['first_R028.png', 0], ['first_R004.png', 2], ['first_R085.png', 2], ['second_R31.png', 2], ['second_R27.png', 1], ['second_R17.png', 1], ['first_R103.png', 1], ['first_R060.png', 2], ['first_R046.png', 1],
                           ['second_R39.png', 0], ['second_R26.png', 2], ['first_R131.png', 1], ['first_R043.png', 0], ['first_R074.png', 1], ['second_R02.png', 1], ['first_R020.png', 2], ['second_R59.png', 0], ['first_R136.png', 1], ['first_R077.png', 1],
                           ['first_R078.png', 2], ['first_R130.png', 1], ['first_R124.png', 0], ['first_R067.png', 1], ['second_R04.png', 0], ['first_R109.png', 1], ['first_R135.png', 1], ['first_R057.png', 0], ['first_R128.png', 0], ['first_R051.png', 1],
                           ['second_R19.png', 0], ['first_R107.png', 1], ['first_R126.png', 1], ['first_R041.png', 0], ['first_R122.png', 1], ['first_R003.png', 1], ['first_R032.png', 1], ['first_R015.png', 1], ['second_R21.png', 1], ['first_R076.png', 1],
                           ['first_R071.png', 1], ['first_R049.png', 0], ['first_R010.png', 0], ['first_R027.png', 1], ['first_R005.png', 0], ['first_R013.png', 1], ['second_R50.png', 0], ['first_R105.png', 0], ['first_R096.png', 1], ['second_R01.png', 1],
                           ['first_R001.png', 1], ['first_R129.png', 2], ['first_R138.png', 2], ['second_R52.png', 1], ['first_R081.png', 1], ['first_R031.png', 1], ['second_R42.png', 2], ['first_R108.png', 0], ['first_R075.png', 1], ['second_R09.png', 0],
                           ['first_R038.png', 0], ['first_R059.png', 1], ['second_R30.png', 1], ['first_R068.png', 1], ['second_R22.png', 2], ['first_R115.png', 0], ['second_R20.png', 1], ['first_R092.png', 0]]
    test_category_list = [['second_R24.png', 1], ['first_R022.png', 1], ['first_R069.png', 0], ['second_R54.png', 1], ['first_R040.png', 0], ['first_R093.png', 1], ['first_R039.png', 1], ['second_R07.png', 1], ['first_R016.png', 1], ['first_R072.png', 1],
                          ['first_R084.png', 0], ['first_R036.png', 0], ['second_R18.png', 0], ['first_R042.png', 2], ['first_R017.png', 1], ['first_R120.png', 1], ['first_R083.png', 2], ['first_R002.png', 1], ['second_R15.png', 2], ['first_R007.png', 2],
                          ['second_R33.png', 1], ['first_R111.png', 2], ['first_R134.png', 2], ['second_R47.png', 1], ['first_R030.png', 1], ['second_R28.png', 2], ['first_R090.png', 1], ['first_R014.png', 1], ['second_R06.png', 1], ['first_R110.png', 1],
                          ['second_R58.png', 1], ['first_R029.png', 0], ['second_R45.png', 1], ['first_R056.png', 1], ['second_R57.png', 2], ['first_R080.png', 0], ['first_R024.png', 1], ['second_R38.png', 2], ['first_R089.png', 1], ['second_R11.png', 2]]

    train_category_map = dict(train_category_list)
    test_category_map = dict(test_category_list)

    def organize_by_category():
        print("\nStep 3: Organizing files into category subdirectories")
        def move_files(source_dir, category_map):
            for category in set(category_map.values()):
                os.makedirs(os.path.join(source_dir, str(category)), exist_ok=True)
            for filename, category in category_map.items():
                src_path = os.path.join(source_dir, filename)
                dest_path = os.path.join(source_dir, str(category), filename)
                if os.path.exists(src_path):
                    shutil.move(src_path, dest_path)
        move_files(train_cc_dir, train_category_map)
        move_files(train_mask_dir, train_category_map)
        move_files(test_cc_dir, test_category_map)
        move_files(test_mask_dir, test_category_map)
        print("Finished organizing files by category.")

    prepare_directories(all_dirs)
    process_first_shot_images()
    split_data()
    organize_by_category()
    
    print("\nDataset preparation complete!")

if __name__ == "__main__":
    main()