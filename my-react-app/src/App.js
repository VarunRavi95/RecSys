import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

import Slider from 'react-slick';
import "slick-carousel/slick/slick.css";
import "slick-carousel/slick/slick-theme.css";

function App() {
    const [query, setQuery] = useState('');
    const [recommendations, setRecommendations] = useState([]);

    const handleSearch = async () => {
        try {
            const response = await axios.post('http://127.0.0.1:5000/recommend', { query });
            setRecommendations(Array.isArray(response.data) ? response.data : []);
        } catch (error) {
            console.error("Error fetching recommendations:", error);
            setRecommendations([]);
        }
    };

    const sliderSettings = {
        dots: true,
        infinite: false,
        speed: 500,
        slidesToShow: 3,
        slidesToScroll: 1,
        responsive: [
            {
                breakpoint: 1024,
                settings: {
                    slidesToShow: 3,
                    slidesToScroll: 1,
                    infinite: true,
                    dots: true
                }
            },
            {
                breakpoint: 600,
                settings: {
                    slidesToShow: 2,
                    slidesToScroll: 1,
                    initialSlide: 2
                }
            },
            {
                breakpoint: 480,
                settings: {
                    slidesToShow: 1,
                    slidesToScroll: 1
                }
            }
        ]
    };

    return (
        <div className="App container mt-5">
            <h1 className="text-center mb-4">Real Estate Listing Recommender</h1>
            <div className="d-flex justify-content-center mb-3">
                <input
                    type="text"
                    value={query}
                    onChange={e => setQuery(e.target.value)}
                    placeholder="Enter your search query"
                    className="form-control w-50 mr-2"
                />
                <button onClick={handleSearch} className="btn btn-primary">Search</button>
            </div>

            <Slider {...sliderSettings}>
                {recommendations.length > 0 ? (
                    recommendations.map((item, index) => (
                        <div key={index} className="card-wrapper">
                            <div className="card">
                                <a href={item.listing_url} target="_blank" rel="noopener noreferrer">
                                    <img src={item.picture_url} className="card-img-top" alt="Listing" />
                                </a>
                                <div className="card-body">
                                    <h5 className="card-title">{item.neighbourhood}</h5>
                                    <p className="card-text"><strong>Description:</strong> {item.description}</p>
                                </div>
                            </div>
                        </div>
                    ))
                ) : (
                    <p className="text-center">No recommendations found</p>
                )}
            </Slider>
        </div>
    );
}

export default App;
