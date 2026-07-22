.PHONY: mac run-mac clean-mac

MAC_TARGET := lucas_kanade_mac
MAC_SOURCE := mac/lucas_kanade_mac.mm
MAC_FRAMEWORKS := -framework CoreFoundation -framework CoreGraphics -framework ImageIO

mac:
	clang++ -std=c++17 -Wall -Wextra -Wpedantic $(MAC_SOURCE) $(MAC_FRAMEWORKS) -o $(MAC_TARGET)

run-mac: mac
	./$(MAC_TARGET)

clean-mac:
	rm -f $(MAC_TARGET) output_mac.png
