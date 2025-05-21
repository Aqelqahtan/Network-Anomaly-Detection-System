print("📦 Network Anomaly Detection System")
print("1. Prepare dataset (split train/test)")
print("2. Train model (Random Forest)")
print("3. Predict and evaluate")
print("4. Evaluate with visuals")


choice = input("Choose an option [1–4]: ")

if choice == "1":
    from utils import load_and_clean_kdd, save_train_test_split
    df = load_and_clean_kdd("data/kddcup99_csv.csv")
    save_train_test_split(df)

elif choice == "2":
    import train

elif choice == "3":
    import predict

elif choice == "4":
    import visualize_results

else:
    print("❌ Invalid choice. Exiting.")
