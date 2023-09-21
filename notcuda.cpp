#include <iostream>
#include "Header.h"
#include <SFML/Graphics.hpp>
#include <tinycolormap.hpp>
#include <random>

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
    autoe.convert_to_virdis();
    autoe.copy_output_gpu2cpu();

    sf::RenderWindow window(sf::VideoMode(windowWidth, windowHeight), "SFML Background Image");
    window.setVerticalSyncEnabled(true);
//    window.setFramerateLimit(1000);

    sf::Texture texture;
    // bind the texture to OpenGL
//    sf::Texture::bind(&texture);
    texture.create(windowWidth, windowHeight);

    // Create an SFML sprite with the texture
    sf::Sprite backSprite(texture);

    sf::Font font;
    if (!font.loadFromFile("arial.ttf")) {
        // Handle font loading error
    }

    sf::Text textBox1;
    sf::Text textBox2;
    sf::Text textBox3;

// Set the font
    textBox1.setFont(font);
    textBox2.setFont(font);
    textBox3.setFont(font);

// Set the text content
    textBox1.setString("Top view");
    textBox2.setString("Gradient");
    textBox3.setString("");

// Set the position
    textBox1.setPosition(10, 5);  // Adjust these coordinates as needed
    textBox2.setPosition(10, 60);  // Adjust these coordinates as needed
    textBox3.setPosition(10, 110);  // Adjust these coordinates as needed

// Set the text color
    textBox1.setFillColor(sf::Color::Black);
    textBox2.setFillColor(sf::Color::Black);
    textBox3.setFillColor(sf::Color::Black);

// Set the character size
    textBox1.setCharacterSize(50);  // Adjust the size as needed
    textBox2.setCharacterSize(50);  // Adjust the size as needed
    textBox3.setCharacterSize(50);  // Adjust the size as needed

    autoe.verbose = false;
    sf::Time deltaTime; sf::Clock clock;
    sf::Time value_change;
    sf::Time compute;
    sf::Time copying;

    bool computation_toggle = true;
    bool view_toggle = true;
    bool run_toggle = true;

    while (window.isOpen()) {
        sf::Event event{};
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed) {
                window.close();
            }
            if (event.type == sf::Event::KeyPressed)
            {
                if (event.key.code == sf::Keyboard::Space)
                {// Space bar is pressed
                    run_toggle = !run_toggle;
                    if (run_toggle){textBox3.setString("");}
                    else {textBox3.setString("PAUSED");}
                }
            }
            if (event.type == sf::Event::MouseButtonPressed)
            {// Check which mouse button was pressed
                if (event.mouseButton.button == sf::Mouse::Left)
                {// Left mouse button was clicked event.mouseButton.x and event.mouseButton.y contain the mouse click coordinates
                    computation_toggle = !computation_toggle;
                    if (computation_toggle){textBox2.setString("Gradient");}
                    else {textBox2.setString("Reachability");}
                }
                if (event.mouseButton.button == sf::Mouse::Right)
                {// Left mouse button was clicked event.mouseButton.x and event.mouseButton.y contain the mouse click coordinates
                    view_toggle = !view_toggle;
                    autoe.switch_zy();
                    if (view_toggle){textBox1.setString("Top view");}
                    else {textBox1.setString("Side view");}
                }
            }
        }

        // Get the mouse position relative to the top-left corner of the window
        sf::Vector2i mousePosition = sf::Mouse::getPosition(window);
        sf::Vector2u windowSize = window.getSize();
        sf::Vector2i windowCenter(windowSize.x / 2, windowSize.y / 2);
        sf::Vector2i relativePosition = mousePosition - windowCenter;

        // Other game logic and updates here

        clock.restart();
        if(run_toggle){
        if (view_toggle){
            autoe.change_z_value((float) relativePosition.y / 2.f);
        } else {
            autoe.change_y_value((float) relativePosition.y / 2.f);
        }}
        value_change = clock.restart();
        if (computation_toggle){
            autoe.dist_to_virdis_pipeline();
        } else {
            autoe.reachability_to_img_pipeline();
        }
        compute = clock.restart();
        autoe.virdisresult_gpu2cpu();
        copying = clock.restart();

        std::cout
        << " value change: " << value_change.asMicroseconds()
        << "\n compute: " << compute.asMicroseconds()
        << "\n copying: " << copying.asMicroseconds()
                << std::endl;
        std::cout << "GPUop total: " << (value_change+compute+copying).asMicroseconds() << " mirco sec" << std::endl;

        clock.restart();

        texture.update(autoe.virdisTexture);

        deltaTime = clock.restart();
        std::cout << "Pixels update: " << deltaTime.asMicroseconds() << " mirco sec\n" << std::endl;

        // Clear the window
        window.clear();

        // Draw the background image
        window.draw(backSprite);
        window.draw(textBox1);
        window.draw(textBox2);
        window.draw(textBox3);

        // Display everything
        window.display();
    }

    autoe.delete_all();
    return 0;
}