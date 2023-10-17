import config


def main():
    print("Running NicheCompass with configuration:")
    config_object = config.Config("test_config.json")
    for key, value in config_object.options.items():
        print(key, ":", value)


if __name__ == '__main__':
    main()
