#include <iostream>
#include "Header.h"
#include <SFML/Graphics.hpp>
#include <tinycolormap.hpp>
#include <random>

int maincpp() {
    std::cout << "hey that's c++" << std::endl;
    thisiscuda();
    return 0;
}
int main_simple(void){
    AutoEstimator autoe{6, 7};
    autoe.compute_dist();
    autoe.copy_output_gpu2cpu();
    autoe.delete_all();
    return 0;
}

int main(void){
    int windowWidth=1920*1;
    int windowHeight=1080*1;
    AutoEstimator autoe{windowWidth, windowHeight};
    autoe.compute_dist();
    autoe.compute_result_norm();
    autoe.copy_output_gpu2cpu();

    sf::RenderWindow window(sf::VideoMode(windowWidth, windowHeight), "SFML Background Image");
    window.setVerticalSyncEnabled(false);
//    window.setFramerateLimit(1000);

    // Create an SFML image
    sf::Image backgroundImage;
    backgroundImage.create(windowWidth, windowHeight);

    tinycolormap::Color color = tinycolormap::GetColor(
            *autoe.result_norm.elements,
            tinycolormap::ColormapType::Viridis);

    Matrix data = autoe.result_norm;

    autoe.verbose = false;
    sf::Time deltaTime; sf::Clock clock;

    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed) {
                window.close();
            }
        }

        // Get the mouse position relative to the top-left corner of the window
        sf::Vector2i mousePosition = sf::Mouse::getPosition(window);
        sf::Vector2u windowSize = window.getSize();
        sf::Vector2i windowCenter(windowSize.x / 2, windowSize.y / 2);
        sf::Vector2i relativePosition = mousePosition - windowCenter;

        // Other game logic and updates here

        clock.restart();

        autoe.change_zvalue(relativePosition.y/2);
        autoe.compute_dist();
        autoe.compute_result_norm();
        autoe.copy_output_gpu2cpu();

        deltaTime = clock.restart();
        std::cout << "compute: " << deltaTime.asMilliseconds() << std::endl;
        clock.restart();
        sf::Color pxcolor;

        // Fill the image with RGB colors
        for (int y = 0; y < windowHeight; ++y) {
            for (int x = 0; x < windowWidth; ++x) {
                int row = y * windowWidth + x;
                float norm = *(data.elements + row)/400;
                color = tinycolormap::GetColor(norm,
                                               tinycolormap::ColormapType::Viridis);

//            float rval = color.r();
//            float gval = color.g();
//            float bval = color.b();
                pxcolor.r = color.ri();
                pxcolor.g = color.gi();
                pxcolor.b = color.bi();
                backgroundImage.setPixel(x, y, pxcolor);
            }
        }

        // Create an SFML texture from the random image
        sf::Texture backGroundTexture;
        backGroundTexture.loadFromImage(backgroundImage);

        // Create an SFML sprite with the random texture
        sf::Sprite backSprite(backGroundTexture);

        deltaTime = clock.restart();
        std::cout << "pixels: " << deltaTime.asMilliseconds() << std::endl;

        // Clear the window
        window.clear();

        // Draw the background image first
        window.draw(backSprite);

        // Draw other game elements here

        // Display everything
        window.display();
    }


    autoe.delete_all();
    return 0;
}