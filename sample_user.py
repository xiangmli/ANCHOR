import json
import os
import random
from typing import Dict, List, Tuple, Any
from collections import defaultdict


class UserSampler:
    """
    Sample users and prepare data for agent training and testing.
    """

    def __init__(
            self,
            base_path: str = r"C:\Users\86156\Desktop\1 Projects\小论文\参考代码\AgentRecBench\process_data\process_data\output_data_all",
            dataset: str = "amazon",
            random_seed: int = 42
    ):
        """
        Initialize the user sampler.

        Args:
            base_path: Base path to the data directory
            dataset: Dataset name (amazon, goodreads, yelp)
            random_seed: Random seed for reproducibility
        """
        self.base_path = base_path
        self.dataset = dataset
        self.dataset_path = os.path.join(base_path, dataset)

        random.seed(random_seed)

        # File paths
        self.user_file = os.path.join(self.dataset_path, 'user.json')
        self.item_file = os.path.join(self.dataset_path, 'item.json')
        self.review_file = os.path.join(self.dataset_path, 'review.json')

        # Data storage
        self.items_dict: Dict[str, Dict] = {}
        self.users_dict: Dict[str, Dict] = {}
        self.user_reviews: Dict[str, List[Dict]] = defaultdict(list)

    def load_data(self) -> None:
        """Load all necessary data files."""
        print("Loading data files...")

        # Load items
        print(f"  Loading items from {self.item_file}...")
        with open(self.item_file, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                item_id = item.get('item_id')
                if item_id:
                    self.items_dict[item_id] = item
        print(f"  Loaded {len(self.items_dict)} items")

        # Load users
        print(f"  Loading users from {self.user_file}...")
        with open(self.user_file, 'r', encoding='utf-8') as f:
            for line in f:
                user = json.loads(line.strip())
                user_id = user.get('user_id')
                if user_id:
                    self.users_dict[user_id] = user
        print(f"  Loaded {len(self.users_dict)} users")

        # Load reviews and group by user
        print(f"  Loading reviews from {self.review_file}...")
        with open(self.review_file, 'r', encoding='utf-8') as f:
            for line in f:
                review = json.loads(line.strip())
                user_id = review.get('user_id')
                if user_id:
                    self.user_reviews[user_id].append(review)

        # Sort reviews by timestamp for each user
        for user_id in self.user_reviews:
            self.user_reviews[user_id].sort(key=lambda x: x.get('timestamp', 0))

        print(f"  Loaded reviews for {len(self.user_reviews)} users")
        print("Data loading complete!\n")

    def sample_users(
            self,
            num_users: int = 100,
            min_reviews: int = 10
    ) -> List[str]:
        """
        Sample users with minimum review count.

        Args:
            num_users: Number of users to sample
            min_reviews: Minimum number of reviews required

        Returns:
            List of sampled user IDs
        """
        print(f"Sampling {num_users} users with at least {min_reviews} reviews...")

        # Filter eligible users
        eligible_users = [
            user_id for user_id, reviews in self.user_reviews.items()
            if len(reviews) >= min_reviews
        ]

        print(f"  Found {len(eligible_users)} eligible users")

        if len(eligible_users) < num_users:
            print(f"  Warning: Only {len(eligible_users)} eligible users available")
            return eligible_users

        sampled_user_ids = random.sample(eligible_users, num_users)
        print(f"  Sampled {len(sampled_user_ids)} users\n")

        return sampled_user_ids

    def format_interaction(
            self,
            review: Dict[str, Any],
            idx: int
    ) -> str:
        """
        Format a single interaction with detailed information.

        Args:
            review: Review dictionary
            idx: Interaction index

        Returns:
            Formatted string
        """
        item_id = review.get('item_id')
        item = self.items_dict.get(item_id, {})

        # Extract item fields
        main_category = item.get('main_category', 'N/A')
        title = item.get('title', 'N/A')
        features = item.get('features', [])
        description = item.get('description', [])
        price = item.get('price', 'N/A')
        categories = item.get('categories', [])

        # Format features and description
        features_str = '; '.join(features) if features else 'N/A'
        description_str = ' '.join(description) if isinstance(description, list) else str(
            description) if description else 'N/A'
        categories_str = ' > '.join(categories) if categories else 'N/A'

        # Truncate long fields
        if len(features_str) > 300:
            features_str = features_str[:297] + '...'
        if len(description_str) > 400:
            description_str = description_str[:397] + '...'

        # Extract review fields
        stars = review.get('stars', 'N/A')
        review_title = review.get('title', 'N/A')
        review_text = review.get('text', 'N/A')

        if len(review_text) > 500:
            review_text = review_text[:497] + '...'

        return f"""--- Interaction {idx} ---
Item ID: {item_id}
Main Category: {main_category}
Title: {title}
Features: {features_str}
Description: {description_str}
Price: {price}
Categories: {categories_str}

User Rating: {stars} stars
Review Title: {review_title}
Review Text: {review_text}
"""

    def format_item_info(
            self,
            item_id: str,
            idx: int
    ) -> str:
        """
        Format item information without review details.

        Args:
            item_id: Item ID
            idx: Item index

        Returns:
            Formatted string
        """
        item = self.items_dict.get(item_id, {})

        main_category = item.get('main_category', 'N/A')
        title = item.get('title', 'N/A')
        features = item.get('features', [])
        description = item.get('description', [])
        price = item.get('price', 'N/A')
        categories = item.get('categories', [])

        features_str = '; '.join(features) if features else 'N/A'
        description_str = ' '.join(description) if isinstance(description, list) else str(
            description) if description else 'N/A'
        categories_str = ' > '.join(categories) if categories else 'N/A'

        if len(features_str) > 300:
            features_str = features_str[:297] + '...'
        if len(description_str) > 400:
            description_str = description_str[:397] + '...'

        return f"""--- Item {idx}: {item_id} ---
Main Category: {main_category}
Title: {title}
Features: {features_str}
Description: {description_str}
Price: {price}
Categories: {categories_str}
"""

    def create_candidate_set(
            self,
            positive_item_id: str,
            user_interacted_items: set,
            num_negatives: int = 19
    ) -> Dict[str, Any]:
        """
        Create a candidate set with negative sampling.

        Args:
            positive_item_id: The ground truth item ID
            user_interacted_items: Set of items the user has interacted with
            num_negatives: Number of negative samples

        Returns:
            Dictionary containing candidate_list and ground_truth
        """
        # Get all available items
        all_items = set(self.items_dict.keys())

        # Negative pool: items user hasn't interacted with
        negative_pool = list(all_items - user_interacted_items)

        if len(negative_pool) < num_negatives:
            print(f"    Warning: Only {len(negative_pool)} negative samples available, requested {num_negatives}")
            num_negatives = len(negative_pool)

        # Sample negative items
        negative_samples = random.sample(negative_pool, num_negatives)

        # Create candidate list
        candidate_list = negative_samples + [positive_item_id]
        random.shuffle(candidate_list)

        return {
            'candidate_list': candidate_list,
            'ground_truth': positive_item_id
        }

    # def prepare_user_data(
    #         self,
    #         user_id: str,
    #         alpha: float = 0.8,
    #         num_negatives: int = 19
    # ) -> Dict[str, Any]:
    #     """
    #     Prepare all data for a single user.
    #
    #     Args:
    #         user_id: User ID
    #         alpha: Ratio of training data for initialization (0-1)
    #         num_negatives: Number of negative samples per positive sample
    #
    #     Returns:
    #         Dictionary containing all prepared data
    #     """
    #     reviews = self.user_reviews[user_id]
    #
    #     # Split: last interaction as test, rest as train
    #     train_reviews = reviews[:-1]
    #     test_review = reviews[-1]
    #
    #     # Split training data
    #     init_size = int(len(train_reviews) * alpha)
    #     init_reviews = train_reviews[:init_size]
    #     refinement_reviews = train_reviews[init_size:]
    #
    #     # Format initialization data (with full details)
    #     init_data = '\n'.join([
    #         self.format_interaction(review, idx)
    #         for idx, review in enumerate(init_reviews, 1)
    #     ])
    #
    #     # Get all interacted items
    #     user_interacted_items = {review.get('item_id') for review in reviews}
    #
    #     # Create candidate sets for refinement data
    #     refinement_candidate_sets = []
    #     for review in refinement_reviews:
    #         positive_item_id = review.get('item_id')
    #         candidate_set = self.create_candidate_set(
    #             positive_item_id,
    #             user_interacted_items,
    #             num_negatives
    #         )
    #
    #         # Format candidate items info
    #         candidate_items_info = '\n'.join([
    #             self.format_item_info(item_id, idx)
    #             for idx, item_id in enumerate(candidate_set['candidate_list'], 1)
    #         ])
    #
    #         refinement_candidate_sets.append({
    #             **candidate_set,
    #             'candidate_items_info': candidate_items_info
    #         })
    #
    #     # Create test candidate set
    #     test_positive_item_id = test_review.get('item_id')
    #     test_candidate_set = self.create_candidate_set(
    #         test_positive_item_id,
    #         user_interacted_items,
    #         num_negatives
    #     )
    #
    #     test_candidate_items_info = '\n'.join([
    #         self.format_item_info(item_id, idx)
    #         for idx, item_id in enumerate(test_candidate_set['candidate_list'], 1)
    #     ])
    #
    #     return {
    #         'user_id': user_id,
    #         'init_data': init_data,
    #         'init_reviews_count': len(init_reviews),
    #         'refinement_candidate_sets': refinement_candidate_sets,
    #         'refinement_count': len(refinement_reviews),
    #         'test_candidate_set': {
    #             **test_candidate_set,
    #             'candidate_items_info': test_candidate_items_info
    #         },
    #         'total_reviews': len(reviews)
    #     }

    def prepare_user_data(
            self,
            user_id: str,
            alpha: float = 0.8,
            num_negatives: int = 19
    ) -> Dict[str, Any]:
        """
        Prepare all data for a single user.

        Args:
            user_id: User ID
            alpha: Ratio of training data for initialization (0-1)
            num_negatives: Number of negative samples per positive sample

        Returns:
            Dictionary containing all prepared data
        """
        reviews = self.user_reviews[user_id]

        # Split: last interaction as test, rest as train
        train_reviews = reviews[:-1]
        test_review = reviews[-1]

        # Split training data
        init_size = int(len(train_reviews) * alpha)
        init_reviews = train_reviews[:init_size]
        refinement_reviews = train_reviews[init_size:]

        # Format initialization data (with full details)
        init_data = '\n'.join([
            self.format_interaction(review, idx)
            for idx, review in enumerate(init_reviews, 1)
        ])

        # Get all interacted items
        user_interacted_items = {review.get('item_id') for review in reviews}

        # Create candidate sets for refinement data
        refinement_candidate_sets = []
        for review in refinement_reviews:
            positive_item_id = review.get('item_id')
            candidate_set = self.create_candidate_set(
                positive_item_id,
                user_interacted_items,
                num_negatives
            )

            # Format candidate items info
            candidate_items_info = '\n'.join([
                self.format_item_info(item_id, idx)
                for idx, item_id in enumerate(candidate_set['candidate_list'], 1)
            ])

            # ===================================================================
            # 新增：提取用户对 ground_truth 商品的评价数据（用于反思机制）
            # ===================================================================
            item_info = self.items_dict.get(positive_item_id, {})

            user_review_data = {
                'stars': review.get('stars'),
                'title': review.get('title', ''),
                'text': review.get('text', ''),
                'item_info': {
                    'main_category': item_info.get('main_category', 'N/A'),
                    'title': item_info.get('title', 'N/A'),
                    'features': item_info.get('features', []),
                    'description': item_info.get('description', []),
                    'price': item_info.get('price', 'N/A'),
                    'categories': item_info.get('categories', [])
                }
            }

            refinement_candidate_sets.append({
                **candidate_set,
                'candidate_items_info': candidate_items_info,
                'user_review': user_review_data  # 添加用户评价数据
            })

        # Create test candidate set (测试集暂时不需要 user_review)
        test_positive_item_id = test_review.get('item_id')
        test_candidate_set = self.create_candidate_set(
            test_positive_item_id,
            user_interacted_items,
            num_negatives
        )

        test_candidate_items_info = '\n'.join([
            self.format_item_info(item_id, idx)
            for idx, item_id in enumerate(test_candidate_set['candidate_list'], 1)
        ])

        return {
            'user_id': user_id,
            'init_data': init_data,
            'init_reviews_count': len(init_reviews),
            'refinement_candidate_sets': refinement_candidate_sets,
            'refinement_count': len(refinement_reviews),
            'test_candidate_set': {
                **test_candidate_set,
                'candidate_items_info': test_candidate_items_info
            },
            'total_reviews': len(reviews)
        }


    def sample_and_prepare(
            self,
            num_users: int = 100,
            min_reviews: int = 10,
            alpha: float = 0.8,
            num_negatives: int = 19,
            output_dir: str = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Complete pipeline: sample users and prepare all their data.

        Args:
            num_users: Number of users to sample
            min_reviews: Minimum reviews per user
            alpha: Initialization data ratio
            num_negatives: Number of negative samples
            output_dir: Base directory to save results (e.g., 'data')

        Returns:
            Dictionary mapping user_id to prepared data
        """
        # Load data
        self.load_data()

        # Sample users
        sampled_user_ids = self.sample_users(num_users, min_reviews)

        # Prepare data for each user
        print("Preparing data for sampled users...")
        all_user_data = {}

        for idx, user_id in enumerate(sampled_user_ids, 1):
            print(f"  Processing user {idx}/{len(sampled_user_ids)}: {user_id}")
            user_data = self.prepare_user_data(user_id, alpha, num_negatives)
            all_user_data[user_id] = user_data

            print(f"    Init reviews: {user_data['init_reviews_count']}")
            print(f"    Refinement sets: {user_data['refinement_count']}")
            print(f"    Test set: 1")

        print(f"\nData preparation complete for {len(all_user_data)} users!")

        # Save to file if output_dir specified
        if output_dir:
            # Create dataset-specific subdirectory
            dataset_dir = os.path.join(output_dir, self.dataset)
            os.makedirs(dataset_dir, exist_ok=True)

            # Save file directly in dataset directory with simple name
            output_file = os.path.join(dataset_dir, 'sampled_users_data.json')

            print(f"\nSaving results to {output_file}...")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(all_user_data, f, ensure_ascii=False, indent=2)
            print("Save complete!")

        return all_user_data


# Usage example
if __name__ == "__main__":
    # Initialize sampler
    sampler = UserSampler(
        dataset="amazon",
        random_seed=42
    )

    # Sample and prepare data
    user_data = sampler.sample_and_prepare(
        num_users=5,  # Sample 10 users for testing
        min_reviews=15,  # Minimum 15 reviews per user
        alpha=0.8,  # 80% for initialization
        num_negatives=9,  # 19 negative samples per positive
        output_dir="data"  # Base directory (will create data/amazon/)
    )

    # Print summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    total_init_interactions = sum(data['init_reviews_count'] for data in user_data.values())
    total_refinement_sets = sum(data['refinement_count'] for data in user_data.values())
    total_test_sets = len(user_data)

    print(f"Total sampled users: {len(user_data)}")
    print(f"Total initialization interactions: {total_init_interactions}")
    print(f"Total refinement candidate sets: {total_refinement_sets}")
    print(f"Total test sets: {total_test_sets}")

    # Example: Access data for first user
    first_user_id = list(user_data.keys())[0]
    first_user_data = user_data[first_user_id]

    print(f"\nExample - User: {first_user_id}")
    print(f"Initialization data preview (first 500 chars):")
    print(first_user_data['init_data'][:500] + "...")
    print(f"\nNumber of refinement candidate sets: {len(first_user_data['refinement_candidate_sets'])}")
    print(f"Test candidate set size: {len(first_user_data['test_candidate_set']['candidate_list'])}")

