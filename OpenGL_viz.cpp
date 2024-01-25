#include "HeaderCPP.h"
#include "HeaderCUDA.h"
#include <SFML/Graphics.hpp>
#include <iostream>
#include <tinycolormap.hpp>
/* #include <random> */
#include <Eigen/Dense>

int main() {
    float pixel_density = 1; // sample/pixel
    float pix_size = 1;      // mm between each screen pixel
    int windowWidth = 1920;
    int windowHeight = 1080;
    std::cout << "let's go!";
    AutoEstimator autoe{(int)(pixel_density * windowWidth),
                        (int)(pixel_density * windowHeight),
                        1 / pix_size / pixel_density};
    sf::RenderWindow window(sf::VideoMode(windowWidth, windowHeight),
                            "SFML Background Image");
    window.setVerticalSyncEnabled(false);
    //    window.setFramerateLimit(1000);

    sf::Texture texture;
    texture.create((int)(pixel_density * windowWidth),
                   (int)(pixel_density * windowHeight));

    // Create an SFML sprite with the texture
    sf::Sprite backSprite(texture);
    backSprite.setScale(1 / pixel_density, 1 / pixel_density);

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
    textBox1.setPosition(10, 5);   // Adjust these coordinates as needed
    textBox2.setPosition(10, 60);  // Adjust these coordinates as needed
    textBox3.setPosition(10, 110); // Adjust these coordinates as needed

    // Set the text color
    textBox1.setFillColor(sf::Color::Black);
    textBox2.setFillColor(sf::Color::Black);
    textBox3.setFillColor(sf::Color::Black);

    // Set the character size
    textBox1.setCharacterSize(50); // Adjust the size as needed
    textBox2.setCharacterSize(50); // Adjust the size as needed
    textBox3.setCharacterSize(50); // Adjust the size as needed

    autoe.verbose = false;
    sf::Time deltaTime;
    sf::Clock clock;
    sf::Time value_change;
    sf::Time compute;
    sf::Time copying;

    bool computation_toggle = true;
    bool view_toggle = true;
    bool run_toggle = true;

    int avg_size = 60 * 1;
    int avg_counter = 0;
    Eigen::VectorXi vect_for_avg = Eigen::VectorXi::Zero(avg_size);

    while (window.isOpen()) {
        sf::Event event{};
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed) {
                window.close();
            }
            if (event.type == sf::Event::KeyPressed) {
                if (event.key.code ==
                    sf::Keyboard::Space) { // Space bar is pressed
                    run_toggle = !run_toggle;
                    if (run_toggle) {
                        textBox3.setString("");
                    } else {
                        textBox3.setString("PAUSED");
                    }
                }
            }
            if (event.type ==
                sf::Event::MouseButtonPressed) { // Check which mouse button was
                                                 // pressed
                if (event.mouseButton.button ==
                    sf::Mouse::Left) { // Left mouse button was clicked
                                       // event.mouseButton.x and
                                       // event.mouseButton.y contain the mouse
                                       // click coordinates
                    computation_toggle = !computation_toggle;
                    if (computation_toggle) {
                        textBox2.setString("reachability_to_img_pipelinef3");
                    } else {
                        textBox2.setString("dist_to_virdis_pipelinef3");
                    }
                }
                if (event.mouseButton.button ==
                    sf::Mouse::Right) { // Left mouse button was clicked
                                        // event.mouseButton.x and
                                        // event.mouseButton.y contain the mouse
                                        // click coordinates
                    view_toggle = !view_toggle;
                    autoe.switch_zy();
                    if (view_toggle) {
                        textBox1.setString("Top view");
                    } else {
                        textBox1.setString("Side view");
                    }
                }
            }
        }

        // Get the mouse position relative to the top-left corner of the window
        sf::Vector2i mousePosition = sf::Mouse::getPosition(window);
        sf::Vector2u windowSize = window.getSize();
        sf::Vector2i windowCenter(windowSize.x / 2, windowSize.y / 2);
        sf::Vector2i relativePosition = (mousePosition - windowCenter) * 2;

        // Other game logic and updates here

        clock.restart();
        if (run_toggle) {
            avg_counter = (avg_counter + 1) % avg_size;
            vect_for_avg[avg_counter] = (int)compute.asMicroseconds();
            textBox3.setString(std::to_string(calculateMedian(vect_for_avg)) +
                               " us");
            if (view_toggle) {
                autoe.change_z_value((float)relativePosition.y / 2.f);
            } else {
                autoe.change_y_value((float)relativePosition.y / 2.f);
            }
        }

        value_change = clock.restart();
        if (computation_toggle) {
            autoe.reachability_to_img_pipelinef3();
            autoe.derivate_output();
            autoe.dist_to_virdis_pipelinef3();
        } else {
            autoe.dist_to_virdis_pipelinef3();
        }

        compute = clock.restart();
        autoe.virdisresult_gpu2cpu();
        copying = clock.restart();

        std::cout << " value change: " << value_change.asMicroseconds()
                  << "\n compute: " << compute.asMicroseconds()
                  << "\n copying: " << copying.asMicroseconds() << std::endl;
        std::cout << "GPUop total: "
                  << (value_change + compute + copying).asMicroseconds()
                  << " mirco sec" << std::endl;

        clock.restart();

        texture.update(autoe.virdisTexture);

        deltaTime = clock.restart();
        std::cout << "Pixels update: " << deltaTime.asMicroseconds()
                  << " mirco sec\n"
                  << std::endl;

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
