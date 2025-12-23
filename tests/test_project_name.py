from metbit.utility import project_name_generator


def test_project_name_generator_format():
    name = project_name_generator()

    assert len(name.split("_", 1)) == 2
    timestamp, _ = name.split("_", 1)
    assert timestamp.isdigit()
    assert len(timestamp) >= 17  # YYYYMMDDHHMMSSmmm
